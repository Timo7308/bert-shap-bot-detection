# ============================================================
# SHAP LOCAL (STRICT) – Token-level SHAP
# Kategorien bleiben: TP/TN/FP/FN × Sicher/Unsicher
# Confidence:
#   Confident   >= 75%
#   Non confident 55–60% 
# ============================================================

import numpy as np
import pandas as pd
import torch
import shap
import re
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from tqdm import tqdm

# -------------------- Device --------------------
device = torch.device("cpu")
print(f"[INFO] Device = {device}")

# -------------------- Paths / Settings --------------------
CSV_PATH    = "/content/twibot3.csv"
MODEL_DIR   = "/content/best/"
TEXT_COL    = "clean_text"
LABEL_COL   = "label"

MAX_LEN     = 128
SAMPLE_SIZE = 500
MAX_EVALS   = 300
TOP_K       = 10

CONF_HIGH = 75.0

# Unsicher: wir picken Beispiele bevorzugt in 55-60
UNSICHER_MIN    = 55.0
UNSICHER_MAX    = 60.0
UNSICHER_TARGET = 59.0

ID2LABEL = {0: "HUMAN", 1: "BOT"}
LABEL2ID = {"human": 0, "bot": 1}

# -------------------- Load data --------------------
df = pd.read_csv(CSV_PATH, sep=None, engine="python")
df = df[[TEXT_COL, LABEL_COL]].dropna()
df[LABEL_COL] = df[LABEL_COL].astype(str).str.lower().map(LABEL2ID)
df = df[df[LABEL_COL].isin([0, 1])].reset_index(drop=True)
print(f"[INFO] Data rows after cleaning: {len(df):,}")

# -------------------- Load model + tokenizer --------------------
tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR, use_fast=True)
print("[INFO] Tokenizer type:", type(tokenizer))

model = AutoModelForSequenceClassification.from_pretrained(
    MODEL_DIR,
    id2label=ID2LABEL,
    label2id={v: k for k, v in ID2LABEL.items()}
)
model.to(device).eval()

# ============================================================
# Prediction function (for SHAP)
# ============================================================
@torch.inference_mode()
def predict_proba(texts):
    if isinstance(texts, str):
        texts = [texts]
    enc = tokenizer(
        list(texts),
        truncation=True,
        padding=True,
        max_length=MAX_LEN,
        return_tensors="pt"
    ).to(device)
    logits = model(**enc).logits
    probs = torch.softmax(logits, dim=-1).detach().cpu().numpy()
    return probs

# ============================================================
# SHAP Explainer (STRICT local)
# ============================================================
masker = shap.maskers.Text(tokenizer)
try:
    explainer = shap.Explainer(predict_proba, masker, algorithm="partition")
except TypeError:
    explainer = shap.Explainer(predict_proba, masker)

def shap_local(text: str):
    base_probs = predict_proba([text])[0]
    pred_idx = int(np.argmax(base_probs))
    pred_label = ID2LABEL[pred_idx]
    pred_percent = float(base_probs[pred_idx] * 100)

    exp = explainer([text], max_evals=MAX_EVALS)

    if exp.values.ndim != 3:
        raise RuntimeError(f"Unexpected SHAP shape: {exp.values.shape} (expected (1, n_tokens, n_classes))")

    # IMPORTANT: SHAP values correspond to exp.data tokens (masker tokens)
    shap_tokens = list(exp.data[0])
    shap_vals   = exp.values[0]  # (n_tokens, 2)

    return base_probs, pred_idx, pred_label, pred_percent, shap_tokens, shap_vals

# ============================================================
# Confidence / Category
# ============================================================
def confidence_band(pred_percent: float) -> str:
    if pred_percent >= CONF_HIGH:
        return "Sicher"
    elif UNSICHER_MIN <= pred_percent <= UNSICHER_MAX:
        return "Unsicher"
    else:
        return "Mittel"

def get_category(true_y: int, pred_y: int, pred_percent: float):
    band = confidence_band(pred_percent)
    if band == "Mittel":
        return None

    if true_y == 1 and pred_y == 1: base = "TP"
    elif true_y == 0 and pred_y == 0: base = "TN"
    elif true_y == 0 and pred_y == 1: base = "FP"
    else: base = "FN"

    return f"{base}_{band}"

# ============================================================
# Surface-level "words/units" from original text
# Keep: hashtags, mentions, URLs, words, numbers; keep punctuation as separators
# ============================================================
SURFACE_RE = re.compile(r"(https?://\S+|www\.\S+|#\w+|@\w+|\w+(?:'\w+)?|\d+(?:[.,]\d+)?|[^\s])", re.UNICODE)

def surface_units(text: str):
    return [m.group(0) for m in SURFACE_RE.finditer(text)]

# ============================================================
# Normalize helpers for greedy matching
# ============================================================
def norm_surface(s: str) -> str:
    # Keep URL/USER/HTTPURL as literal markers
    t = s.strip()
    if t.lower().startswith("http") or t.lower().startswith("www."):
        return "httpurl"
    return re.sub(r"[^a-z0-9#@]+", "", t.lower())

def norm_token(t: str) -> str:
    if t is None:
        return ""
    t = str(t)

    # remove common specials
    if t in {"<s>", "</s>", "<pad>"}:
        return ""

    # normalize SHAP/masker tokens:
    # - RoBERTa marker
    if t.startswith("Ġ"):
        t = t[1:]
    # - BPE continuation marker (if present in outputs)
    t = re.sub(r"@@$", "", t)

    # map common placeholders
    if t.upper() == "HTTPURL":
        return "httpurl"
    if t.upper() == "USER":
        return "user"

    # strip non-alnum but keep #/@
    return re.sub(r"[^a-z0-9#@]+", "", t.lower())

# ============================================================
# Token-SHAP -> surface-unit aggregation (approx.)
# Greedy: consume consecutive tokens until their normalized concat matches the unit (or is a good prefix)
# If matching fails, fallback assigns one token to the unit to avoid infinite loops.
# ============================================================
def project_shap_to_surface_words(text: str, shap_tokens, shap_vals_2d):
    units = surface_units(text)
    units_norm = [norm_surface(u) for u in units]

    # Prepare token stream (normalized + keep original token for debugging if needed)
    toks = []
    vals = []
    for tok, sv in zip(shap_tokens, shap_vals_2d):
        nt = norm_token(tok)
        if nt == "":
            continue
        toks.append(nt)
        vals.append(sv)

    # Greedy consume
    out_rows = []
    ti = 0
    nT = len(toks)

    for u, un in zip(units, units_norm):
        if un == "":
            continue

        # If we run out of tokens, stop
        if ti >= nT:
            break

        start_ti = ti
        acc = None
        built = ""

        # Try to build up to a match
        # Limit how many tokens we try per surface unit to avoid runaway
        for _ in range(12):
            if ti >= nT:
                break

            built_next = built + toks[ti]
            # initialize accumulator
            if acc is None:
                acc = vals[ti].copy()
            else:
                acc = acc + vals[ti]

            # accept token
            built = built_next
            ti += 1

            # exact match -> done
            if built == un:
                break

            # if built is already longer and no prefix relation, we can stop early
            # (but keep what we have, since it's the best we can do)
            if not un.startswith(built) and not built.startswith(un):
                break

        # Fallback: if we didn't consume anything (shouldn't happen), consume 1 token
        if acc is None:
            acc = vals[ti].copy()
            built = toks[ti]
            ti += 1

        out_rows.append({
            "Word": u,  # keep original surface form
            "SHAP_HUMAN": float(acc[0]),
            "SHAP_BOT": float(acc[1]),
        })

        # Safety: if we made zero progress (should not), force progress
        if ti == start_ti:
            ti += 1

    dfw = pd.DataFrame(out_rows)
    if len(dfw) == 0:
        return pd.DataFrame(columns=["Word","SHAP_HUMAN","SHAP_BOT","DELTA_BOT_MINUS_HUMAN"])

    # Aggregate duplicates (same surface word may occur multiple times)
    dfw = dfw.groupby("Word", as_index=False)[["SHAP_HUMAN","SHAP_BOT"]].sum()
    dfw["DELTA_BOT_MINUS_HUMAN"] = dfw["SHAP_BOT"] - dfw["SHAP_HUMAN"]
    return dfw

# ============================================================
# Sample + Predict for category selection
# ============================================================
df_sample = df.sample(min(SAMPLE_SIZE, len(df)), random_state=42).reset_index(drop=True)

preds_num, preds_lbl, preds_pct = [], [], []
probs_h, probs_b = [], []
cats = []

for text, true_y in tqdm(zip(df_sample[TEXT_COL], df_sample[LABEL_COL]), total=len(df_sample), desc="Predict"):
    probs = predict_proba(text)[0]
    pred_y = int(np.argmax(probs))
    pred_percent = float(probs[pred_y] * 100)

    preds_num.append(pred_y)
    preds_lbl.append(ID2LABEL[pred_y])
    preds_pct.append(pred_percent)
    probs_h.append(float(probs[0] * 100))
    probs_b.append(float(probs[1] * 100))
    cats.append(get_category(int(true_y), pred_y, pred_percent))

df_sample["pred_label_num"] = preds_num
df_sample["pred_label"] = preds_lbl
df_sample["pred_percent"] = preds_pct
df_sample["prob_human"] = probs_h
df_sample["prob_bot"] = probs_b
df_sample["category"] = cats

df_candidates = df_sample.dropna(subset=["category"]).reset_index(drop=True)

print("\n[INFO] Category counts (Sicher >=75, Unsicher 55-60):")
print(df_candidates["category"].value_counts())

# ============================================================
# Exactly 1 example per category (8 total)
# Sicher: max pred_percent
# Unsicher: prefer near UNSICHER_TARGET within 55-60
# ============================================================
CATEGORIES = [
    "TP_Sicher","TP_Unsicher","TN_Sicher","TN_Unsicher",
    "FP_Sicher","FP_Unsicher","FN_Sicher","FN_Unsicher"
]

picked_rows, missing = [], []

for c in CATEGORIES:
    sub = df_candidates[df_candidates["category"] == c]
    if len(sub) == 0:
        missing.append(c)
        continue

    if c.endswith("_Sicher"):
        idx = sub["pred_percent"].idxmax()
    else:
        # already restricted by get_category to 55-60, but keep target selection
        idx = (sub["pred_percent"] - UNSICHER_TARGET).abs().idxmin()

    picked_rows.append(df_candidates.loc[idx])

if missing:
    print("\n[WARN] Missing categories:", ", ".join(missing))
    print("[WARN] Consider increasing SAMPLE_SIZE or widening Unsicher band slightly.")

examples = pd.DataFrame(picked_rows).reset_index(drop=True)
print(f"\n[INFO] Selected examples: {len(examples)} / 8")

# ============================================================
# SHAP Local Analysis + Save
# ============================================================
SAVE_PATH = "/content/shap_local_results.txt"

with open(SAVE_PATH, "w", encoding="utf-8") as f:
    f.write("Lokale SHAP-Analyse (STRICT) – approx. Wortebene durch Projektion auf Surface-Units\n")
    f.write(f"Tokenizer type: {type(tokenizer)}\n")
    f.write(f"Confidence: Sicher >= {CONF_HIGH:.1f}% | Unsicher in [{UNSICHER_MIN:.1f},{UNSICHER_MAX:.1f}]% (Target {UNSICHER_TARGET:.1f}%)\n")
    f.write(f"NOTE: Word-level is an approximation (token-level SHAP projected onto surface units).\n")
    f.write("="*120 + "\n\n")

    for i, row in tqdm(examples.iterrows(), total=len(examples), desc="SHAP"):
        text = row[TEXT_COL]
        true_label = ID2LABEL[int(row[LABEL_COL])]

        base_probs, pred_idx, pred_label, pred_percent, shap_tokens, shap_vals = shap_local(text)

        # Project token SHAP -> surface words/units
        df_words = project_shap_to_surface_words(text, shap_tokens, shap_vals)

        top_bot = (
            df_words[df_words["DELTA_BOT_MINUS_HUMAN"] > 0]
            .sort_values("DELTA_BOT_MINUS_HUMAN", ascending=False)
            .head(TOP_K)
        )
        top_hum = (
            df_words[df_words["DELTA_BOT_MINUS_HUMAN"] < 0]
            .sort_values("DELTA_BOT_MINUS_HUMAN", ascending=True)
            .head(TOP_K)
        )

        f.write(f"Beispiel {i+1}/{len(examples)}\n")
        f.write(f"Kategorie: {row['category']}\n")
        f.write(f"True: {true_label} | Pred: {pred_label} ({pred_percent:.1f}%)\n")
        f.write(f"Probs: HUMAN={base_probs[0]*100:.1f}% | BOT={base_probs[1]*100:.1f}%\n\n")

        f.write("Text:\n")
        f.write(text + "\n\n")

        f.write("Top Wörter/Units Richtung BOT (delta = SHAP_BOT - SHAP_HUMAN):\n")
        if len(top_bot) == 0:
            f.write("(keine positiven delta-Units gefunden)\n")
        else:
            for _, r in top_bot.iterrows():
                f.write(f"{r['Word']}: delta={r['DELTA_BOT_MINUS_HUMAN']:.6f} | bot={r['SHAP_BOT']:.6f} | human={r['SHAP_HUMAN']:.6f}\n")

        f.write("\nTop Wörter/Units Richtung HUMAN (delta = SHAP_BOT - SHAP_HUMAN):\n")
        if len(top_hum) == 0:
            f.write("(keine negativen delta-Units gefunden)\n")
        else:
            for _, r in top_hum.iterrows():
                f.write(f"{r['Word']}: delta={r['DELTA_BOT_MINUS_HUMAN']:.6f} | bot={r['SHAP_BOT']:.6f} | human={r['SHAP_HUMAN']:.6f}\n")

        f.write("\n" + "-"*120 + "\n\n")

print(f"[INFO] SHAP results saved to: {SAVE_PATH}")




