# =============================
# Compare different variants @ 20k (strict 50/50), 80/10/10
# Setup: global Seed, Author-Cap=5, NUM_EPOCHS=4, LR=3e-5

# Purpose of this script:
# - Compare multiple transformer model variants for bot-vs-human classification under the same conditions.
# - Use a fixed training size (20k tweets, strict 50/50), a fixed author-disjoint 80/10/10 split,
#   and the same training setup (epochs, learning rate, author-cap) for all models.
#
# Key idea for a fair model comparison:
# - All models are trained and evaluated on the same data split and sampling strategy.
# - Differences in performance are therefore mainly attributable to the model choice,
#   not to different data or split conditions.
#
# Important note:
# - This is a controlled comparison for robustness/selection, not hyperparameter tuning.
# - Results depend on this dataset, preprocessing pipeline, and the chosen random seed.
# =============================

import os, random, numpy as np, pandas as pd, torch, matplotlib.pyplot as plt, tempfile, shutil
from sklearn.model_selection import GroupShuffleSplit
from sklearn.metrics import accuracy_score, f1_score, classification_report
from transformers import (
    AutoTokenizer, AutoModelForSequenceClassification, TrainingArguments, Trainer,
    DataCollatorWithPadding, EarlyStoppingCallback
)

# -------- Repro (globaler Seed wie Single-Run) --------
SEED = 42
def set_seed(s=SEED):
    random.seed(s); np.random.seed(s); torch.manual_seed(s)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(s)

set_seed(SEED)
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Device:", DEVICE)

MASTER_PATH   = "twibot2.csv"
VAL_FRAC_OVERALL  = 0.10
TEST_FRAC_OVERALL = 0.10

TRAIN_SIZE    = 20000   
MAX_TWEETS_PER_AUTHOR = 5
MAX_LEN       = 128
NUM_EPOCHS    = 4     
BATCH         = 32
GRAD_ACC      = 2
LR            = 3e-5
WARMUP_STEPS  = 200
WEIGHT_DECAY  = 0.01
LABEL_SMOOTH  = 0.0
PATIENCE      = 2

bf16_ok = torch.cuda.is_available() and torch.cuda.get_device_capability(0)[0] >= 8
fp16_ok = torch.cuda.is_available() and not bf16_ok

# -------- Modell-List (Encoder) --------
MODEL_IDS = [
    "vinai/bertweet-base",               # BERTweet (Twitter)
    "roberta-base",                      # RoBERTa
    "microsoft/deberta-v3-base",         # DeBERTa-v3
    "microsoft/MiniLM-L12-H384-uncased", # MiniLM
    "chandar-lab/NeoBERT",               # NeoBERT (trust_remote_code)
]
REMOTE_CODE_MODELS = {"chandar-lab/NeoBERT"}

# -------- Load data --------
def load_df(path):
    df = pd.read_csv(path)[["clean_text","label","author_id"]].dropna()
    df["label"] = df["label"].map({"human":0, "bot":1})
    df["author_id"] = df["author_id"].astype(str)
    df = df.drop_duplicates(subset=["clean_text"]).reset_index(drop=True)
    return df

df_master = load_df(MASTER_PATH)
print(f"Master (nach Drop-Dups): {len(df_master):,} | H={(df_master.label==0).sum()} | B={(df_master.label==1).sum()}")

# -------- Author-disjoint Splits --------
def author_split(df, test_size=0.20):
    gss = GroupShuffleSplit(n_splits=1, test_size=test_size, random_state=SEED)
    idx_tr, idx_te = next(gss.split(df, groups=df["author_id"]))
    tr, te = df.iloc[idx_tr].reset_index(drop=True), df.iloc[idx_te].reset_index(drop=True)
    assert set(tr.author_id).isdisjoint(set(te.author_id))
    return tr, te

def group_split(df, val_size=0.20):
    gss = GroupShuffleSplit(n_splits=1, test_size=val_size, random_state=SEED)
    idx_tr, idx_val = next(gss.split(df, groups=df["author_id"]))
    tr = df.iloc[idx_tr].reset_index(drop=True)
    va = df.iloc[idx_val].reset_index(drop=True)
    assert set(tr.author_id).isdisjoint(set(va.author_id))
    return tr, va

def cap_per_author(df, k=MAX_TWEETS_PER_AUTHOR):
    # wie im Single-Run: max k Tweets pro Autor im Trainingspool
    return (df.groupby("author_id", group_keys=False)
              .apply(lambda g: g.sample(n=min(len(g), k), random_state=SEED))
              .reset_index(drop=True))

def sample_balanced(df, size):
    # strict 50/50 Sampling aus dem (gecappten) Train-Pool
    need = size // 2
    tr_h = df[df.label==0].sample(n=need, random_state=SEED)
    tr_b = df[df.label==1].sample(n=need, random_state=SEED)
    return pd.concat([tr_h, tr_b]).sample(frac=1, random_state=SEED).reset_index(drop=True)

# 1) TEST abspalten (author-disjoint)
train_pool, test_df = author_split(df_master, test_size=TEST_FRAC_OVERALL)

# 2) VAL aus dem verbleibenden Train-Pool so wählen, dass global 10% Val entstehen
adj_val_in_train = VAL_FRAC_OVERALL / (1.0 - TEST_FRAC_OVERALL)  # 0.10 / 0.90
dev_train_full, val_df = group_split(train_pool, val_size=adj_val_in_train)

# 3) Cap pro Autor im DEV-Train-Pool (wie Single-Run)
dev_train_capped = cap_per_author(dev_train_full, MAX_TWEETS_PER_AUTHOR)

print(f"Pools: TrainPool pre-cap={len(dev_train_full):,} | post-cap={len(dev_train_capped):,} | "
      f"Val={len(val_df):,} | Test={len(test_df):,}")

# 4) strict 50/50 Trainings-Subset (20k) aus dem capped Train-Pool
need = TRAIN_SIZE // 2
if (dev_train_capped.label==0).sum() < need or (dev_train_capped.label==1).sum() < need:
    raise ValueError(f"Nicht genug Daten im capped Train-Pool für 50/50 @ {TRAIN_SIZE}")

train_df = sample_balanced(dev_train_capped, TRAIN_SIZE)
print(f"Train {len(train_df)} (H={(train_df.label==0).sum()}, B={(train_df.label==1).sum()}) | "
      f"Val {len(val_df)} | Test {len(test_df)}")

# -------- Tokenizer/DS Helpers --------
def get_tokenizer(model_id):
    if model_id.startswith("vinai/bertweet"):
        return AutoTokenizer.from_pretrained(model_id, normalization=True, use_fast=True)
    elif model_id in REMOTE_CODE_MODELS:
        return AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
    else:
        return AutoTokenizer.from_pretrained(model_id, use_fast=True)

def build_ds(tokenizer, texts, labels, max_len):
    enc = tokenizer(texts, truncation=True, max_length=max_len)
    class DS(torch.utils.data.Dataset):
        def __init__(self, e, y): self.e=e; self.y=y
        def __len__(self): return len(self.y)
        def __getitem__(self, i):
            item = {k: torch.tensor(v[i]) for k,v in self.e.items()}
            item["labels"] = torch.tensor(self.y[i], dtype=torch.long)
            return item
    return DS(enc, list(labels))

def compute_metrics(p):
    preds = np.argmax(p.predictions, axis=1)
    return {"accuracy": accuracy_score(p.label_ids, preds),
            "f1_macro": f1_score(p.label_ids, preds, average="macro")}

# -------- TrainingArguments: bestes Checkpoint nach Val-F1 laden --------
def make_args(out_dir):
    base = dict(
        output_dir=out_dir,
        save_strategy="epoch",            # speichern pro Epoche
        save_total_limit=1,               # nur bestes halten
        load_best_model_at_end=True,      # bestes (Val-F1) nach Training laden
        metric_for_best_model="f1_macro",
        greater_is_better=True,

        # Eval/Logging
        logging_strategy="epoch",
        report_to="none",

        # Train
        per_device_train_batch_size=BATCH,
        per_device_eval_batch_size=BATCH,
        gradient_accumulation_steps=GRAD_ACC,
        num_train_epochs=NUM_EPOCHS,
        learning_rate=LR,
        weight_decay=WEIGHT_DECAY,
        warmup_steps=WARMUP_STEPS,
        label_smoothing_factor=LABEL_SMOOTH,
        lr_scheduler_type="cosine",

        # Mixed precision & Loader
        bf16=bf16_ok, fp16=fp16_ok, seed=SEED,
        dataloader_num_workers=2,
        disable_tqdm=True,
    )
    # Kompatibilität eval_strategy vs evaluation_strategy
    fields = getattr(TrainingArguments, "__dataclass_fields__", {})
    if "eval_strategy" in fields: base["eval_strategy"] = "epoch"
    else: base["evaluation_strategy"] = "epoch"
    return TrainingArguments(**base)

# -------- Vergleich: Training & Evaluation je Modell --------
all_results = []
for mid in MODEL_IDS:
    print(f"\n====================\nModel: {mid}\n====================")
    tokenizer = get_tokenizer(mid)

    if mid in REMOTE_CODE_MODELS:
        model = AutoModelForSequenceClassification.from_pretrained(
            mid, num_labels=2, trust_remote_code=True
        ).to(DEVICE)
    else:
        model = AutoModelForSequenceClassification.from_pretrained(
            mid, num_labels=2
        ).to(DEVICE)

    data_collator = DataCollatorWithPadding(
        tokenizer, pad_to_multiple_of=8 if torch.cuda.is_available() else None
    )

    # temporäres Verzeichnis (bestes Checkpoint wird gespeichert & am Ende gelöscht)
    tmp_dir = tempfile.mkdtemp(prefix=f"cmp_{mid.split('/')[-1]}_20k_seed{SEED}_")
    args = make_args(out_dir=tmp_dir)

    try:
        trainer = Trainer(
            model=model, args=args,
            train_dataset=build_ds(tokenizer, train_df.clean_text.tolist(), train_df.label.tolist(), MAX_LEN),
            eval_dataset =build_ds(tokenizer, val_df.clean_text.tolist(),   val_df.label.tolist(),   MAX_LEN),
            compute_metrics=compute_metrics,
            callbacks=[EarlyStoppingCallback(early_stopping_patience=PATIENCE)],
            data_collator=data_collator,
        )
        trainer.train()  # lädt danach automatisch das beste Val-F1-Checkpoint in den Speicher

        # Test (immer gleich, author-disjoint)
        test_ds = build_ds(tokenizer, test_df.clean_text.tolist(), test_df.label.tolist(), MAX_LEN)
        preds = trainer.predict(test_ds)
        y_true = preds.label_ids
        y_pred = np.argmax(preds.predictions, axis=1)

        rep = classification_report(y_true, y_pred, target_names=["Human","Bot"], output_dict=True)
        row = {
            "model": mid,
            "accuracy": rep["accuracy"],
            "f1_macro": rep["macro avg"]["f1-score"],
            "f1_human": rep["Human"]["f1-score"],
            "f1_bot":   rep["Bot"]["f1-score"],
        }
        all_results.append(row)
    finally:
        # Temp-Ordner aufräumen
        try: shutil.rmtree(tmp_dir, ignore_errors=True)
        except Exception: pass

# -------- Zusammenfassung & Plot --------
res_df = pd.DataFrame(all_results).sort_values("f1_macro", ascending=False)
print("\n==== Vergleich @ 20k (80/10/10, bestes Checkpoint, fixer TEST, cap=5, 1 Seed) ====")
print(res_df.to_string(index=False))

plt.figure(figsize=(10,6))
x = np.arange(len(res_df))
plt.bar(x - 0.25, res_df["f1_macro"], width=0.25, label="Macro-F1")
plt.bar(x,          res_df["f1_human"], width=0.25, label="F1 Human")
plt.bar(x + 0.25,   res_df["f1_bot"],   width=0.25, label="F1 Bot")
plt.xticks(x, [m.split("/")[-1] for m in res_df["model"]], rotation=20)
plt.ylim(0.45, 0.75)
plt.ylabel("F1")
plt.title("Vergleich @ 20k (80/10/10, strict 50/50, cap=5, Seed=42, bestes Checkpoint)")
plt.grid(axis="y", alpha=0.3)
plt.legend()
plt.tight_layout()
plt.show()

