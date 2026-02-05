# =======================
# BERTweet-base
# Datasize: 20.000
# Split: 80/10/10 (author-disjoint)
# =======================

import os, random, tempfile
import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt
import os, zipfile

from typing import Tuple
from sklearn.model_selection import GroupShuffleSplit
from sklearn.metrics import classification_report, confusion_matrix

from transformers import (
    AutoTokenizer, AutoModelForSequenceClassification,
    TrainingArguments, Trainer, DataCollatorWithPadding,
    EarlyStoppingCallback
)

MASTER_PATH = "twibot3.csv"
SCHEME_TAG = "80-10-10"
VAL_FRAC_OVERALL, TEST_FRAC_OVERALL = 0.10, 0.10
TRAIN_SIZE = 20000
SEED = 42


NUM_EPOCHS, PATIENCE = 4, 2
BATCH, FIX_MAX_LEN = 32, 128
BASE_LR = 2e-5
MAX_TWEETS_PER_AUTHOR = 5
GRAD_ACC = 2
WEIGHT_DECAY = 0.01
WARMUP_STEPS = 200
LABEL_SMOOTHING = 0.0

def set_seed(s=SEED):
    random.seed(s)
    np.random.seed(s)
    torch.manual_seed(s)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(s)

set_seed(SEED)


DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
try:
    if torch.cuda.is_available():
        torch.set_float32_matmul_precision("medium")
except Exception:
    pass
print("Device:", DEVICE)


import transformers, sklearn
print("Versions:",
      "transformers", transformers.__version__,
      "sklearn", sklearn.__version__,
      "torch", torch.__version__,
      "numpy", np.__version__,
      "pandas", pd.__version__)


bf16_ok = torch.cuda.is_available() and torch.cuda.get_device_capability(0)[0] >= 8
fp16_ok = torch.cuda.is_available() and not bf16_ok
DL_WORKERS = max(2, min(8, (os.cpu_count() or 4)//2))

def plot_cm_green_red(cm, labels=("Human", "Bot"), title="Confusion Matrix (counts)"):
    """
    Overlays two heatmaps with transparency:
      - diagonal (correct) in Greens
      - off-diagonal (wrong) in Reds
    FIX: mask zeros so only non-zero cells are drawn (no color wash-over)
    """
    cm = np.asarray(cm)

    # Diagonal (correct) and off-diagonal (wrong)
    diag = np.zeros_like(cm)
    np.fill_diagonal(diag, np.diag(cm))
    off = cm - diag

    # Mask zeros -> transparent
    diag_masked = np.ma.masked_where(diag == 0, diag)
    off_masked  = np.ma.masked_where(off == 0, off)

    fig, ax = plt.subplots(figsize=(6, 6))

    # Wrong first (red), correct on top (green)
    ax.imshow(off_masked,  cmap="Reds",   interpolation="nearest")
    ax.imshow(diag_masked, cmap="Greens", interpolation="nearest")

    ax.set_xticks(range(len(labels)))
    ax.set_yticks(range(len(labels)))
    ax.set_xticklabels(labels)
    ax.set_yticklabels(labels)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title(title)

    # annotate values
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, str(int(cm[i, j])), ha="center", va="center", fontsize=12)

    plt.tight_layout()
    plt.show()

def plot_f1_vs_error_over_epochs(ep_df, title="Validation Macro-F1 vs Error over Epochs"):
    """
    Left axis: Macro-F1
    Right axis: Error = 1 - Accuracy
    """
    if ep_df is None or len(ep_df) == 0:
        print("No epoch dataframe available to plot.")
        return

    ep_plot = ep_df.copy()
    ep_plot["val_error"] = 1.0 - ep_plot["val_accuracy"]

    fig, ax1 = plt.subplots(figsize=(7.8, 4.8))
    ax1.plot(ep_plot["epoch"], ep_plot["val_f1"], marker="o", linewidth=2)
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Validation Macro-F1")
    ax1.grid(alpha=0.3)

    ax2 = ax1.twinx()
    ax2.plot(ep_plot["epoch"], ep_plot["val_error"], marker="s", linewidth=2)
    ax2.set_ylabel("Validation Error (1 - Accuracy)")

    ax1.set_title(title)
    plt.tight_layout()
    plt.show()

def plot_f1_vs_loss_over_epochs(ep_df, title="Validation Macro-F1 vs Loss over Epochs"):
    """
    Left axis: Macro-F1
    Right axis: Val loss
    """
    if ep_df is None or len(ep_df) == 0:
        print("No epoch dataframe available to plot.")
        return

    fig, ax1 = plt.subplots(figsize=(7.8, 4.8))
    ax1.plot(ep_df["epoch"], ep_df["val_f1"], marker="o", linewidth=2)
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Validation Macro-F1")
    ax1.grid(alpha=0.3)

    ax2 = ax1.twinx()
    ax2.plot(ep_df["epoch"], ep_df["val_loss"], marker="s", linewidth=2)
    ax2.set_ylabel("Validation Loss")

    ax1.set_title(title)
    plt.tight_layout()
    plt.show()


def load_df(path):
    df = pd.read_csv(path)[["clean_text","label","author_id"]].dropna()
    df["label"] = df["label"].map({"human":0, "bot":1})
    df["author_id"] = df["author_id"].astype(str)
    df = df.drop_duplicates(subset=["clean_text"]).reset_index(drop=True)
    return df

def author_split(df: pd.DataFrame, test_size: float = TEST_FRAC_OVERALL) -> Tuple[pd.DataFrame, pd.DataFrame]:
    gss = GroupShuffleSplit(n_splits=1, test_size=test_size, random_state=SEED)
    idx_tr, idx_te = next(gss.split(df, groups=df["author_id"]))
    tr = df.iloc[idx_tr].reset_index(drop=True)
    te = df.iloc[idx_te].reset_index(drop=True)
    assert set(tr.author_id).isdisjoint(set(te.author_id))
    return tr, te

def group_split(df: pd.DataFrame, val_size: float = None) -> Tuple[pd.DataFrame, pd.DataFrame]:
    if val_size is None:
        val_size = VAL_FRAC_OVERALL / (1.0 - TEST_FRAC_OVERALL)  # 0.10/0.90
    gss = GroupShuffleSplit(n_splits=1, test_size=val_size, random_state=SEED)
    idx_tr, idx_val = next(gss.split(df, groups=df["author_id"]))
    tr = df.iloc[idx_tr].reset_index(drop=True)
    va = df.iloc[idx_val].reset_index(drop=True)
    assert set(tr.author_id).isdisjoint(set(va.author_id))
    return tr, va

def cap_per_author(df, k):
    return (df.groupby("author_id", group_keys=False)
              .apply(lambda g: g.sample(n=min(len(g), k), random_state=SEED))
              .reset_index(drop=True))

def sample_balanced(df, size):
    need = size // 2
    tr_h = df[df.label==0].sample(n=need, random_state=SEED)
    tr_b = df[df.label==1].sample(n=need, random_state=SEED)
    return pd.concat([tr_h, tr_b]).sample(frac=1, random_state=SEED).reset_index(drop=True)

def compute_metrics(p):
    from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
    preds = np.argmax(p.predictions, axis=1)
    return {
        "accuracy": accuracy_score(p.label_ids, preds),
        "f1_macro": f1_score(p.label_ids, preds, average="macro"),
        "precision_macro": precision_score(p.label_ids, preds, average="macro", zero_division=0),
        "recall_macro":    recall_score(p.label_ids, preds, average="macro",  zero_division=0),
    }

def make_args_tmp(tmp_dir):
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

        bf16=bf16_ok, fp16=fp16_ok,
        seed=SEED,

        dataloader_num_workers=DL_WORKERS,
        dataloader_pin_memory=True,
        dataloader_prefetch_factor=2,
        disable_tqdm=True,
    )
    fields = getattr(TrainingArguments, "__dataclass_fields__", {})
    if "eval_strategy" in fields:
        base["eval_strategy"] = "epoch"
    else:
        base["evaluation_strategy"] = "epoch"
    return TrainingArguments(**base)

def describe_split(df_master, train_full, val, test, train_capped):
    def cnt(df): return len(df), float(df["label"].mean()) if len(df) > 0 else float("nan")
    def acnt(df): return df["author_id"].nunique()

    print("\n--- Split-Transparenz ---")
    print("Tweets (N, Bot-Anteil):",
          "Total", cnt(df_master),
          "| Train pre-cap", cnt(train_full),
          "| Train post-cap", cnt(train_capped),
          "| Val", cnt(val),
          "| Test", cnt(test))
    print("Autoren (#):",
          "Total", acnt(df_master),
          "| Train pre-cap", acnt(train_full),
          "| Train post-cap", acnt(train_capped),
          "| Val", acnt(val),
          "| Test", acnt(test))
    print("-------------------------\n")

df_master = load_df(MASTER_PATH)

train_pool, test_df = author_split(df_master, test_size=TEST_FRAC_OVERALL)

adj_val = VAL_FRAC_OVERALL / (1.0 - TEST_FRAC_OVERALL)
dev_train_full, dev_val = group_split(train_pool, val_size=adj_val)

dev_train = cap_per_author(dev_train_full, MAX_TWEETS_PER_AUTHOR)

n_total = len(df_master)
print(f"[{SEED} | {SCHEME_TAG}] Global: "
      f"Train≈{len(dev_train_full)/n_total:.3f} | "
      f"Val≈{len(dev_val)/n_total:.3f} | "
      f"Test≈{len(test_df)/n_total:.3f}")

describe_split(df_master, dev_train_full, dev_val, test_df, dev_train)

# Check: reicht capped-Train für 20k (balanciert)?
n_h = int((dev_train.label == 0).sum())
n_b = int((dev_train.label == 1).sum())
max_balanced = 2 * min(n_h, n_b)
max_size = min(len(dev_train), max_balanced)
assert TRAIN_SIZE <= max_size, f"TRAIN_SIZE={TRAIN_SIZE} > max_size={max_size} aus dem capped Train."


tokenizer = AutoTokenizer.from_pretrained("vinai/bertweet-base", normalization=True, use_fast=True)

def build_ds(enc, labels):
    class DS(torch.utils.data.Dataset):
        def __init__(self, e, y): self.e = e; self.y = y
        def __len__(self): return len(self.y)
        def __getitem__(self, i):
            item = {k: torch.tensor(v[i]) for k, v in self.e.items()}
            item["labels"] = torch.tensor(self.y[i], dtype=torch.long)
            return item
    return DS(enc, list(labels))

val_enc  = tokenizer(dev_val.clean_text.tolist(),  truncation=True, max_length=FIX_MAX_LEN)
test_enc = tokenizer(test_df.clean_text.tolist(), truncation=True, max_length=FIX_MAX_LEN)
ds_val   = build_ds(val_enc,  dev_val.label.tolist())
ds_test  = build_ds(test_enc, test_df.label.tolist())

collator = DataCollatorWithPadding(
    tokenizer,
    pad_to_multiple_of=8 if torch.cuda.is_available() else None
)

tr_df = sample_balanced(dev_train, TRAIN_SIZE)
tr_enc = tokenizer(tr_df.clean_text.tolist(), truncation=True, max_length=FIX_MAX_LEN)
ds_tr  = build_ds(tr_enc, tr_df.label.tolist())

tmp_dir = tempfile.mkdtemp(prefix=f"run_{SCHEME_TAG}_size{TRAIN_SIZE}_seed{SEED}_")
args = make_args_tmp(tmp_dir)

model = AutoModelForSequenceClassification.from_pretrained(
    "vinai/bertweet-base", num_labels=2
).to(DEVICE)

trainer = Trainer(
    model=model,
    args=args,
    train_dataset=ds_tr,
    eval_dataset=ds_val,
    compute_metrics=compute_metrics,
    callbacks=[EarlyStoppingCallback(early_stopping_patience=PATIENCE)],
    data_collator=collator,
)

trainer.train()

logs = pd.DataFrame(trainer.state.log_history)
for c in ["epoch","eval_loss","eval_f1_macro","eval_accuracy","eval_precision_macro","eval_recall_macro"]:
    if c not in logs.columns:
        logs[c] = np.nan

ep = logs.loc[logs["eval_loss"].notna(),
              ["epoch","eval_loss","eval_f1_macro","eval_precision_macro","eval_recall_macro","eval_accuracy"]].copy()
ep["epoch"] = ep["epoch"].round().astype(int)
ep = ep.groupby("epoch", as_index=False).agg(
    val_loss=("eval_loss","last"),
    val_f1=("eval_f1_macro","last"),
    val_precision=("eval_precision_macro","last"),
    val_recall=("eval_recall_macro","last"),
    val_accuracy=("eval_accuracy","last"),
)

print("\n==== Epochen (Validation) — Single-Seed ====")
epoch_table = ep.copy().round(4)
print(epoch_table.to_string(index=False))

plot_f1_vs_error_over_epochs(ep, title="Validation Macro-F1 vs Error (1-Accuracy) over Epochs")
plot_f1_vs_loss_over_epochs(ep, title="Validation Macro-F1 vs Loss over Epochs")

best_dir = os.path.join(tmp_dir, "best")
os.makedirs(best_dir, exist_ok=True)
trainer.save_model(best_dir)
tokenizer.save_pretrained(best_dir)

epoch_table_path = os.path.join(tmp_dir, "epoch_metrics.csv")
epoch_table.to_csv(epoch_table_path, index=False)
print("\nSaved epoch table to:", epoch_table_path)



zip_path = os.path.join(tmp_dir, "bertweet_best.zip")

def zip_folder(folder_path: str, zip_file_path: str):
    folder_path = os.path.abspath(folder_path)
    with zipfile.ZipFile(zip_file_path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        for root, _, files in os.walk(folder_path):
            for f in files:
                full_path = os.path.join(root, f)
                rel_path = os.path.relpath(full_path, start=os.path.dirname(folder_path))
                zf.write(full_path, rel_path)

zip_folder(best_dir, zip_path)
print("Created ZIP:", zip_path)
print("ZIP size (MB):", round(os.path.getsize(zip_path) / (1024*1024), 2))


test_pred = trainer.predict(ds_test)
y_true = test_pred.label_ids
y_pred = np.argmax(test_pred.predictions, axis=1)

cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
print("\n==== Confusion Matrix (Counts, Test, Single-Seed) ====")
print(pd.DataFrame(cm, index=["True Human","True Bot"], columns=["Pred Human","Pred Bot"]).to_string())

plot_cm_green_red(cm, labels=("Human","Bot"), title=f"Confusion Matrix — Test (seed={SEED})")

rep = classification_report(y_true, y_pred, target_names=["Human","Bot"], output_dict=True)

final_table = pd.DataFrame([
    {"class": "Human", "precision": rep["Human"]["precision"], "recall": rep["Human"]["recall"],
     "f1": rep["Human"]["f1-score"], "support": int(rep["Human"]["support"])},
    {"class": "Bot",   "precision": rep["Bot"]["precision"],   "recall": rep["Bot"]["recall"],
     "f1": rep["Bot"]["f1-score"],   "support": int(rep["Bot"]["support"])},
    {"class": "macro avg", "precision": rep["macro avg"]["precision"], "recall": rep["macro avg"]["recall"],
     "f1": rep["macro avg"]["f1-score"], "support": int(rep["macro avg"]["support"])},
    {"class": "weighted avg", "precision": rep["weighted avg"]["precision"], "recall": rep["weighted avg"]["recall"],
     "f1": rep["weighted avg"]["f1-score"], "support": int(rep["weighted avg"]["support"])},
])

final_table[["precision","recall","f1"]] = final_table[["precision","recall","f1"]].astype(float).round(4)

print("\n==== Test-Metriken (Single-Seed) ====")
print(final_table.to_string(index=False))

final_table_path = os.path.join(tmp_dir, "test_metrics.csv")
final_table.to_csv(final_table_path, index=False)
print("\nSaved test table to:", final_table_path)

print("\nArtifacts directory (kept):", tmp_dir)
print("Best model:", best_dir)

