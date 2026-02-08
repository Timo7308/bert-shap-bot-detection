# =======================
# Bot-vs-Human
# Compare 80/10/10 vs 70/15/15

# Purpose of this script:
# - Train and evaluate a bot-vs-human classifier (BERTweet) on different training set sizes.
# - Compare two dataset split schemes (80/10/10 vs 70/15/15) while keeping authors disjoint
#   between train/validation/test to prevent leakage from the same account.
#
# Key idea for a fair split comparison:
# - Both schemes share the same fixed validation set (Val10) and the same fixed test set (Test10).
# - This allows a direct comparison of model performance across split schemes on identical data.
#
# Important note:
# - Results describe performance for this dataset, preprocessing, and random seed(s).
# - The goal is methodological comparison (split and data size effects), not hyperparameter optimization.

# =======================

import os, random, numpy as np, pandas as pd, torch, matplotlib.pyplot as plt, tempfile, shutil
from sklearn.model_selection import GroupShuffleSplit
from sklearn.metrics import classification_report
from transformers import (
    AutoTokenizer, AutoModelForSequenceClassification, TrainingArguments, Trainer,
    DataCollatorWithPadding, EarlyStoppingCallback
)

# ===== Repro & Speed =====
GLOBAL_SEED = 42
def set_seed(s=GLOBAL_SEED):
    random.seed(s); np.random.seed(s); torch.manual_seed(s)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(s)
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark = True

set_seed(GLOBAL_SEED)
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
if torch.cuda.is_available():
    try:
        torch.set_float32_matmul_precision("medium")
    except Exception:
        pass
print("Device:", DEVICE)

# ===== Pfade & Größen =====
MASTER_PATH = "twibot2.csv"
REQUESTED_SIZES = [2000, 5000, 10000, 20000, 30000, 40000, 50000]

# ===== Trainings-Config =====
NUM_EPOCHS       = 4
WARMUP_STEPS     = 200
WEIGHT_DECAY     = 0.01
GRAD_ACC         = 2
BATCH            = 32
FIX_MAX_LEN      = 128
BASE_LR          = 3e-5
LABEL_SMOOTHING  = 0.0
PATIENCE         = 2
MAX_TWEETS_PER_AUTHOR = 5
SEEDS = [GLOBAL_SEED] 

DL_WORKERS = max(2, min(8, (os.cpu_count() or 4) // 2))
bf16_ok = torch.cuda.is_available() and torch.cuda.get_device_capability(0)[0] >= 8
fp16_ok = torch.cuda.is_available() and not bf16_ok

# ===== Utils =====
def load_df(path):
    df = pd.read_csv(path)[["clean_text","label","author_id"]].dropna()
    df["label"] = df["label"].map({"human":0, "bot":1})
    df["author_id"] = df["author_id"].astype(str)
    df = df.drop_duplicates(subset=["clean_text"]).reset_index(drop=True)
    return df

def group_split(df, test_size, seed=GLOBAL_SEED):
    """Author-disjoint split: returns (train_part, test_part)."""
    gss = GroupShuffleSplit(n_splits=1, test_size=test_size, random_state=seed)
    idx_tr, idx_te = next(gss.split(df, groups=df["author_id"]))
    tr = df.iloc[idx_tr].reset_index(drop=True)
    te = df.iloc[idx_te].reset_index(drop=True)
    assert set(tr.author_id).isdisjoint(set(te.author_id))
    return tr, te

def assert_disjoint(*dfs):
    sets = [set(d.author_id) for d in dfs]
    for i in range(len(sets)):
        for j in range(i+1, len(sets)):
            assert sets[i].isdisjoint(sets[j])

def cap_per_author(df, k):
    return (df.groupby("author_id", group_keys=False)
              .apply(lambda g: g.sample(n=min(len(g), k), random_state=GLOBAL_SEED))
              .reset_index(drop=True))

def sample_balanced(df, size):
    need = size // 2
    tr_h = df[df.label==0].sample(n=need, random_state=GLOBAL_SEED)
    tr_b = df[df.label==1].sample(n=need, random_state=GLOBAL_SEED)
    return pd.concat([tr_h, tr_b]).sample(frac=1, random_state=GLOBAL_SEED).reset_index(drop=True)

def compute_metrics(p):
    from sklearn.metrics import accuracy_score, f1_score
    preds = np.argmax(p.predictions, axis=1)
    return {
        "accuracy": accuracy_score(p.label_ids, preds),
        "f1_macro": f1_score(p.label_ids, preds, average="macro")
    }

def build_ds_enc(enc, labels):
    class DS(torch.utils.data.Dataset):
        def __init__(self, e, y): self.e=e; self.y=y
        def __len__(self): return len(self.y)
        def __getitem__(self, i):
            item = {k: torch.tensor(v[i]) for k,v in self.e.items()}
            item["labels"] = torch.tensor(self.y[i], dtype=torch.long)
            return item
    return DS(enc, list(labels))

# ===== Tokenizer & Collator =====
tokenizer = AutoTokenizer.from_pretrained("vinai/bertweet-base", normalization=True, use_fast=True)
data_collator = DataCollatorWithPadding(tokenizer, pad_to_multiple_of=8 if torch.cuda.is_available() else None)

def tokenize_fast(texts):
    return tokenizer(texts, truncation=True, max_length=FIX_MAX_LEN)

# ===== TrainingArguments =====
def make_args_temp(tmp_dir):
    base = dict(
        output_dir=tmp_dir,
        save_strategy="epoch",
        save_total_limit=1,
        load_best_model_at_end=True,   
        metric_for_best_model="f1_macro",
        greater_is_better=True,

        logging_strategy="epoch",
        report_to="none",

        per_device_train_batch_size=BATCH,
        per_device_eval_batch_size=BATCH,
        gradient_accumulation_steps=GRAD_ACC,
        num_train_epochs=NUM_EPOCHS,
        learning_rate=BASE_LR,
        weight_decay=WEIGHT_DECAY,
        warmup_steps=WARMUP_STEPS,
        label_smoothing_factor=LABEL_SMOOTHING,
        lr_scheduler_type="cosine",

        bf16=bf16_ok, fp16=fp16_ok, seed=GLOBAL_SEED,
        dataloader_num_workers=DL_WORKERS,
        dataloader_pin_memory=True,
        dataloader_prefetch_factor=2,
        disable_tqdm=True,
    )
    fields = getattr(TrainingArguments, "__dataclass_fields__", {})
    if "eval_strategy" in fields: base["eval_strategy"] = "epoch"
    else: base["evaluation_strategy"] = "epoch"
    return TrainingArguments(**base)

# =======================
# FIXED BASE SPLIT (80/10/10) + derive 70/15/15 from Train80
# =======================
def make_fixed_splits(df):
    """
    Creates author-disjoint base split:
      test10 (10%), val10 (10%), train80 (80%)
    Then derives additional 5% val/test from train80 to form 70/15/15:
      test_extra5 (5%), val_extra5 (5%), train70 (70%)

    Important for FAIR COMPARISON over Val-F1:
      - val_common (=val10) is identical in BOTH schemes
      - test_common (=test10) is identical in BOTH schemes
    """
    n_total = len(df)

    # 1) test10 from full
    rest90, test10 = group_split(df, test_size=0.10, seed=GLOBAL_SEED)

    # 2) val10 from remaining 90% -> needs 10% overall = 10/90 of rest90
    val10_frac_in_rest = 0.10 / 0.90
    train80, val10 = group_split(rest90, test_size=val10_frac_in_rest, seed=GLOBAL_SEED)

    assert_disjoint(train80, val10, test10)

    # 3) derive test_extra5 from train80: 5% overall = 5/80 of train80
    test_extra_frac_in_train80 = 0.05 / 0.80  # 0.0625
    train75, test_extra5 = group_split(train80, test_size=test_extra_frac_in_train80, seed=GLOBAL_SEED)

    # 4) derive val_extra5 from train75: 5% overall = 5/75 of train75
    val_extra_frac_in_train75 = 0.05 / 0.75   # 0.066666...
    train70, val_extra5 = group_split(train75, test_size=val_extra_frac_in_train75, seed=GLOBAL_SEED)

    assert_disjoint(train70, val10, val_extra5, test10, test_extra5)

    scheme_80 = {
        "scheme": "80-10-10",
        "train": train80,
        "val":   val10,
        "test":  test10,
        "val_common": val10,
        "test_common": test10,
    }

    scheme_70 = {
        "scheme": "70-15-15",
        "train": train70,
        "val":   pd.concat([val10, val_extra5]).reset_index(drop=True),
        "test":  pd.concat([test10, test_extra5]).reset_index(drop=True),
        "val_common": val10,         # identical to scheme_80
        "test_common": test10,       # identical to scheme_80
        "val_extra": val_extra5,
        "test_extra": test_extra5,
    }

    def frac(x): return len(x)/n_total
    print("\n=== Fixed base split fractions (tweet-level counts; may deviate due to author grouping) ===")
    print(f"Train80: {frac(train80):.3f} | Val10: {frac(val10):.3f} | Test10: {frac(test10):.3f}")
    print("=== Derived scheme fractions (tweet-level counts) ===")
    print(f"Train70: {frac(train70):.3f} | Val15: {frac(scheme_70['val']):.3f} | Test15: {frac(scheme_70['test']):.3f}")

    return {"80-10-10": scheme_80, "70-15-15": scheme_70}

def prepare_scheme_data(scheme_dict):
    """
    - Cap only in training
    - Determine feasible balanced train sizes
    - Build datasets:
        ds_val_schema    -> for early stopping / best checkpoint selection
        ds_val_common    -> for FAIR split comparison (identical Val10)
        ds_test_schema   -> reporting
        ds_test_common   -> optional (identical Test10) for fair test comparison
    """
    train_df_full = scheme_dict["train"]
    val_schema_df = scheme_dict["val"]
    test_schema_df = scheme_dict["test"]
    val_common_df = scheme_dict["val_common"]
    test_common_df = scheme_dict["test_common"]

    # cap only in train
    train_df = cap_per_author(train_df_full, MAX_TWEETS_PER_AUTHOR)

    # feasible balanced sizes
    n_h = int((train_df.label==0).sum()); n_b = int((train_df.label==1).sum())
    max_balanced = 2 * min(n_h, n_b)
    max_size = min(len(train_df), max_balanced)
    sizes = [s for s in REQUESTED_SIZES if s <= max_size]

    # build val (schema) dataset (used by Trainer for eval + early stopping)
    val_schema_enc = tokenizer(val_schema_df.clean_text.tolist(), truncation=True, max_length=FIX_MAX_LEN)
    ds_val_schema = build_ds_enc(val_schema_enc, val_schema_df.label.tolist())

    # build val_common dataset (used for fair comparison only)
    val_common_enc = tokenizer(val_common_df.clean_text.tolist(), truncation=True, max_length=FIX_MAX_LEN)
    ds_val_common = build_ds_enc(val_common_enc, val_common_df.label.tolist())

    # build test datasets
    test_schema_enc = tokenizer(test_schema_df.clean_text.tolist(), truncation=True, max_length=FIX_MAX_LEN)
    ds_test_schema = build_ds_enc(test_schema_enc, test_schema_df.label.tolist())

    test_common_enc = tokenizer(test_common_df.clean_text.tolist(), truncation=True, max_length=FIX_MAX_LEN)
    ds_test_common = build_ds_enc(test_common_enc, test_common_df.label.tolist())

    return train_df, ds_val_schema, ds_val_common, ds_test_schema, ds_test_common, sizes

def train_once(size, seed, ds_val_schema, ds_val_common, ds_test_schema, ds_test_common, train_df, scheme_tag):
    set_seed(seed)

    tr_df = sample_balanced(train_df, size)
    ds_tr = build_ds_enc(tokenize_fast(tr_df.clean_text.tolist()), tr_df.label.tolist())

    model = AutoModelForSequenceClassification.from_pretrained(
        "vinai/bertweet-base", num_labels=2
    ).to(DEVICE)

    tmp_dir = tempfile.mkdtemp(prefix=f"runs_fast_{scheme_tag}_size{size}_seed{seed}_")
    args = make_args_temp(tmp_dir)

    try:
        trainer = Trainer(
            model=model, args=args,
            train_dataset=ds_tr, eval_dataset=ds_val_schema,
            compute_metrics=compute_metrics,
            callbacks=[EarlyStoppingCallback(early_stopping_patience=PATIENCE)],
            data_collator=data_collator,
        )
        trainer.train()  # loads best checkpoint according to schema-specific val_f1_macro

        # (A) schema val report (same as selection criterion)
        val_pred = trainer.predict(ds_val_schema)
        val_rep = classification_report(
            val_pred.label_ids, np.argmax(val_pred.predictions, axis=1),
            target_names=["Human","Bot"], output_dict=True
        )

        # (B) FAIR split comparison val report (identical Val10 for both schemes)
        valc_pred = trainer.predict(ds_val_common)
        valc_rep = classification_report(
            valc_pred.label_ids, np.argmax(valc_pred.predictions, axis=1),
            target_names=["Human","Bot"], output_dict=True
        )

        # (C) schema test report
        test_pred = trainer.predict(ds_test_schema)
        test_rep = classification_report(
            test_pred.label_ids, np.argmax(test_pred.predictions, axis=1),
            target_names=["Human","Bot"], output_dict=True
        )

        # (D) optional: common test (identical Test10 for both schemes)
        testc_pred = trainer.predict(ds_test_common)
        testc_rep = classification_report(
            testc_pred.label_ids, np.argmax(testc_pred.predictions, axis=1),
            target_names=["Human","Bot"], output_dict=True
        )

    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)

    return {
        "scheme": scheme_tag,
        "size": size,
        "seed": seed,

        # schema val (selection criterion)
        "val_f1_macro": float(val_rep["macro avg"]["f1-score"]),
        "val_accuracy": float(val_rep["accuracy"]),
        "val_f1_human": float(val_rep["Human"]["f1-score"]),
        "val_f1_bot":   float(val_rep["Bot"]["f1-score"]),

        # FAIR comparison val (identical across schemes)
        "val_common_f1_macro": float(valc_rep["macro avg"]["f1-score"]),
        "val_common_accuracy": float(valc_rep["accuracy"]),

        # schema test
        "test_f1_macro": float(test_rep["macro avg"]["f1-score"]),
        "test_accuracy": float(test_rep["accuracy"]),
        "test_f1_human": float(test_rep["Human"]["f1-score"]),
        "test_f1_bot":   float(test_rep["Bot"]["f1-score"]),

        # optional: common test (identical across schemes)
        "test_common_f1_macro": float(testc_rep["macro avg"]["f1-score"]),
        "test_common_accuracy": float(testc_rep["accuracy"]),
    }

# =======================
# RUN
# =======================
df_master = load_df(MASTER_PATH)
splits = make_fixed_splits(df_master)

SCHEMES_TO_RUN = ["70-15-15", "80-10-10"]

all_runs = []
for scheme_tag in SCHEMES_TO_RUN:
    print(f"\n========== SCHEMA {scheme_tag} ==========")

    train_df, ds_val_schema, ds_val_common, ds_test_schema, ds_test_common, sizes = prepare_scheme_data(splits[scheme_tag])
    print("Laufe Größen:", sizes)

    for size in sizes:
        print(f"\n===== TRAINING FÜR GRÖSSE {size} ({scheme_tag}) =====")
        for seed in SEEDS:
            res = train_once(
                size=size, seed=seed,
                ds_val_schema=ds_val_schema,
                ds_val_common=ds_val_common,
                ds_test_schema=ds_test_schema,
                ds_test_common=ds_test_common,
                train_df=train_df,
                scheme_tag=scheme_tag
            )
            all_runs.append(res)
            print(
                f"  [{scheme_tag}] "
                f"Val(schema) F1={res['val_f1_macro']:.4f} | "
                f"Val(common) F1={res['val_common_f1_macro']:.4f} | "
                f"Test(schema) F1={res['test_f1_macro']:.4f}"
            )

# =======================
# REPORTING
# =======================
df_runs = pd.DataFrame(all_runs).sort_values(["scheme","size","seed"]).reset_index(drop=True)

# The key column for your split comparison over validation:
#   val_common_f1_macro  (same Val10 for both schemes)
display_cols = [
    "scheme","size","seed",
    "val_common_f1_macro","val_common_accuracy",
    "val_f1_macro","val_accuracy",
    "test_common_f1_macro","test_common_accuracy",
    "test_f1_macro","test_accuracy"
]

pd.set_option("display.max_rows", 200)
pd.set_option("display.width", 160)

print("\n==== Einzelne Durchläufe (alle Schemata) ====")
print(df_runs[display_cols].round(4).to_string(index=False))

# Aggregation
agg = (df_runs
       .groupby(["scheme","size"], as_index=False)
       .agg(
            val_common_f1_mean=("val_common_f1_macro","mean"),
            val_common_f1_std =("val_common_f1_macro","std"),
            val_schema_f1_mean=("val_f1_macro","mean"),
            test_common_f1_mean=("test_common_f1_macro","mean"),
            test_schema_f1_mean=("test_f1_macro","mean"),
       )
      )

print("\n==== Mittelwerte (Split-Vergleich primär über Val10: val_common_f1_mean) ====")
print(agg.round(4).to_string(index=False))

# Pivot for quick view
pivot_val_common = agg.pivot(index="size", columns="scheme", values="val_common_f1_mean").sort_index()
print("\n==== Pivot: Val(common, Val10) Macro-F1 (Mittel) – Größen × Schemata ====")
print(pivot_val_common.round(4).to_string())

# Plot: compare schemes on common validation
fig, ax = plt.subplots(figsize=(9.2, 5.4))
for scheme_tag in df_runs["scheme"].unique():
    sub = (df_runs[df_runs["scheme"]==scheme_tag]
           .groupby("size", as_index=False)["val_common_f1_macro"].mean()
           .sort_values("size"))
    ax.plot(sub["size"].tolist(), sub["val_common_f1_macro"].tolist(),
            marker="o", linewidth=2.0, label=f"{scheme_tag} (Val10 common)")

ax.set_xlabel("Train size (tweets, balanced 50/50, author-disjoint)", fontsize=12)
ax.set_ylabel("Macro-F1 (common validation Val10)", fontsize=12)
ax.set_title("Split comparison via common validation (Val10)", fontsize=13)
ax.set_xticks(sorted(df_runs["size"].unique()))
ax.set_xticklabels([f"{s//1000}k" if s>=1000 else str(s) for s in sorted(df_runs["size"].unique())])
ax.grid(alpha=0.3)
ax.legend(loc="best")
plt.tight_layout()
plt.show()

# Optional: per-scheme Val vs Test plot (schema-specific sets)
for scheme_tag in df_runs["scheme"].unique():
    sub = df_runs[df_runs["scheme"]==scheme_tag].sort_values("size")
    sizes = sub["size"].tolist()
    val_schema_f1 = sub["val_f1_macro"].tolist()
    test_schema_f1 = sub["test_f1_macro"].tolist()

    fig, ax = plt.subplots(figsize=(9.2, 5.4))
    ax.plot(sizes, val_schema_f1, marker="o", linewidth=2.0, label="Val(schema) F1 (selection)")
    ax.plot(sizes, test_schema_f1, marker="s", linewidth=2.0, label="Test(schema) F1 (report)")
    ax.set_xlabel("Train size (tweets, balanced 50/50, author-disjoint)", fontsize=12)
    ax.set_ylabel("Macro-F1", fontsize=12)
    ax.set_title(f"Val(schema) vs Test(schema) – {scheme_tag}", fontsize=13)
    ax.set_xticks(sizes)
    ax.set_xticklabels([f"{s//1000}k" if s>=1000 else str(s) for s in sizes])
    ax.grid(alpha=0.3)
    ax.legend(loc="best")
    plt.tight_layout()
    plt.show()
