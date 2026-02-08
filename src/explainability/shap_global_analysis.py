# =========================================================
# SHAP GLOBAL – mean(|SHAP_class|) je Klasse
# Konsistent zu SHAP local:
#  - Masker = Modell-Tokenizer
#  - SHAP strikt token-level
#  - Projektion token->surface-units via greedy string matching (Approximation)
#  - Ranking global: mean absolute SHAP per class (HUMAN vs BOT)

# Purpose of this script:
# - Select a balanced set of tweets (same number of human and bot examples).
# - Use the trained language model to explain its decisions with SHAP.
# - Aggregate explanations over many tweets to identify words that are most influential
#   for predicting "human" versus "bot".
#
# Important note:
# - The results describe tendencies in this specific sample and model.
# - They do not represent general or causal rules about human or bot behavior.
# =========================================================


# -------------------- Configuration --------------------
CSV_PATH   = "/content/twibot3.csv"
MODEL_DIR  = "/content/best/"

TEXT_COL   = "clean_text"
LABEL_COL  = "label"
AUTHOR_COL = "author_id"   

# Sampling / SHAP
N_PER_CLASS   = 1250        
MIN_LEN       = 5
SEED          = 42

MAX_EVALS     = 120        
BATCH_SIZE    = 16
SHAP_MAX_LEN  = 96           

TOP_K         = 10
EXCLUSIVE_TOPK = True     

INCLUDE_URLS_IN_SURFACE = True

# Export
EXPORT_ORDERED_IDS_CSV = "/content/shap_global_sample_author_ids_ordered.csv"
EXPORT_UNIQUE_IDS_CSV  = "/content/shap_global_sample_author_ids_unique.csv"
EXPORT_SAMPLE_ROWS_CSV = "/content/shap_global_sample_rows.csv"  # optional


# -------------------- Imports & Repro --------------------
import time, random
import numpy as np
import pandas as pd
import torch
import shap
import re
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from IPython.display import display
from tqdm import tqdm

random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)
if torch.cuda.is_available(): torch.cuda.manual_seed_all(SEED)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("[INFO] Device:", device)


# -------------------- CSV robust --------------------
def _read_csv_any(path):
    try:    return pd.read_csv(path, sep=None, engine="python")
    except: return pd.read_csv(path)

df = _read_csv_any(CSV_PATH)
need = {TEXT_COL, LABEL_COL, AUTHOR_COL}
assert need <= set(df.columns), f"Spalten fehlen: {need - set(df.columns)}"

df = df[[TEXT_COL, LABEL_COL, AUTHOR_COL]].dropna(subset=[TEXT_COL, AUTHOR_COL]).copy()
df[TEXT_COL]   = df[TEXT_COL].astype(str)
df[AUTHOR_COL] = df[AUTHOR_COL].astype(str)

# labels -> 0/1
lab = df[LABEL_COL].astype(str).str.strip().str.lower().map({"human":0, "bot":1})
if lab.isna().any():
    lab2 = pd.to_numeric(df[LABEL_COL], errors="coerce")
    lab = lab.fillna(lab2)
df[LABEL_COL] = lab.fillna(0).astype(int)
df = df[df[LABEL_COL].isin([0,1])].reset_index(drop=True)

ID2LABEL = {0:"HUMAN", 1:"BOT"}
LABEL2ID = {"HUMAN":0, "BOT":1}


# -------------------- Modell & Tokenizer --------------------
model = AutoModelForSequenceClassification.from_pretrained(
    MODEL_DIR, id2label=ID2LABEL, label2id=LABEL2ID
).to(device).eval()

# WICHTIG: Masker = Modell-Tokenizer (konsistent)
tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR, use_fast=True)
print("[INFO] Tokenizer is_fast:", getattr(tokenizer, "is_fast", False))

USE_FP16 = torch.cuda.is_available()
if USE_FP16:
    try: model.half()
    except Exception: USE_FP16 = False
print("[INFO] FP16:", USE_FP16)


# -------------------- Model callable (logits oder probs) --------------------
def _encode(texts):
    if isinstance(texts, (np.ndarray, tuple)): texts = list(texts)
    elif isinstance(texts, str): texts = [texts]
    else: texts = list(texts)
    enc = tokenizer(
        texts,
        truncation=True,
        padding=True,
        max_length=128,
        return_tensors="pt"
    )
    return {k: v.to(device) for k, v in enc.items()}

@torch.no_grad()
def f_model_logits(texts):
    enc = _encode(texts)
    if USE_FP16 and device.type == "cuda":
        with torch.cuda.amp.autocast():
            logits = model(**enc).logits
    else:
        logits = model(**enc).logits
    return logits.detach().cpu().numpy()

# -------------------- Balanced Sampling (mit author_ids) --------------------
def sample_balanced(df, n_per_class, min_len, seed):
    bots = df[(df[LABEL_COL]==1) & (df[TEXT_COL].str.len()>=min_len)]
    hums = df[(df[LABEL_COL]==0) & (df[TEXT_COL].str.len()>=min_len)]
    n_b = min(n_per_class, len(bots))
    n_h = min(n_per_class, len(hums))
    if n_b == 0 or n_h == 0:
        raise RuntimeError("Nicht genug Daten je Klasse. Reduziere N_PER_CLASS oder MIN_LEN.")
    sb = bots.sample(n=n_b, random_state=seed)[[TEXT_COL, LABEL_COL, AUTHOR_COL]].copy()
    sh = hums.sample(n=n_h, random_state=seed)[[TEXT_COL, LABEL_COL, AUTHOR_COL]].copy()
    samp = pd.concat([sb, sh], axis=0).sample(frac=1, random_state=seed).reset_index(drop=True)
    return (
        samp[TEXT_COL].tolist(),
        samp[LABEL_COL].tolist(),
        samp[AUTHOR_COL].tolist(),
        samp
    )

texts_global, y_true_global, author_ids_global, sample_df = sample_balanced(df, N_PER_CLASS, MIN_LEN, SEED)
print(f"[INFO] Global sample: {len(texts_global)} texts (BOT={sum(y_true_global)}, HUMAN={len(y_true_global)-sum(y_true_global)})")

# -------------------- SHAP Explainer --------------------
masker = shap.maskers.Text(tokenizer)
explainer = shap.Explainer(f_model_logits, masker, output_names=["HUMAN","BOT"])
# Hinweis: output_names muss zur Logit-Reihenfolge passen; bei ID2LABEL/LABEL2ID i.d.R. ok.

# -------------------- Surface units + Projektion (wie local) --------------------
# Keep: URLs/USER/hashtags/words/numbers (approx. Wortebene)
if INCLUDE_URLS_IN_SURFACE:
    SURFACE_RE = re.compile(r"(https?://\S+|www\.\S+|HTTPURL|USER|#\w+|@\w+|\w+(?:'\w+)?|\d+(?:[.,]\d+)?|[^\s])", re.UNICODE)
else:
    SURFACE_RE = re.compile(r"(HTTPURL|USER|#\w+|@\w+|\w+(?:'\w+)?|\d+(?:[.,]\d+)?|[^\s])", re.UNICODE)

def surface_units(text: str):
    return [m.group(0) for m in SURFACE_RE.finditer(text or "")]

def norm_surface(s: str) -> str:
    t = (s or "").strip()
    tl = t.lower()
    # normalize placeholders
    if tl == "httpurl" or tl.startswith("http") or tl.startswith("www."):
        return "httpurl"
    if tl == "user":
        return "user"
    # keep #/@, remove other noise
    return re.sub(r"[^a-z0-9#@]+", "", tl)

def norm_token(t: str) -> str:
    if t is None:
        return ""
    t = str(t)

    # ignore specials
    if t in {"<s>", "</s>", "<pad>"}:
        return ""

    # RoBERTa whitespace marker
    if t.startswith("Ġ"):
        t = t[1:]

    tl = t.lower()

    # map common placeholders
    if tl == "httpurl":
        return "httpurl"
    if tl == "user":
        return "user"

    return re.sub(r"[^a-z0-9#@]+", "", tl)

def project_token_shap_to_surface(text: str, shap_tokens, shap_vals_2d):
    """
    shap_tokens: list of tokens (exp.data[0])
    shap_vals_2d: array (n_tokens, 2) for [HUMAN, BOT]
    returns df with columns: Word, SHAP_HUMAN, SHAP_BOT
    """
    units = surface_units(text)
    units_norm = [norm_surface(u) for u in units]

    toks, vals = [], []
    for tok, sv in zip(shap_tokens, shap_vals_2d):
        nt = norm_token(tok)
        if nt == "":
            continue
        toks.append(nt)
        vals.append(sv)

    out = []
    ti = 0
    nT = len(toks)

    for u, un in zip(units, units_norm):
        if un == "":
            continue
        if ti >= nT:
            break

        start_ti = ti
        built = ""
        acc = None

        # greedy consume up to 12 tokens
        for _ in range(12):
            if ti >= nT:
                break
            built_next = built + toks[ti]
            if acc is None:
                acc = np.array(vals[ti], dtype=float)
            else:
                acc = acc + np.array(vals[ti], dtype=float)

            built = built_next
            ti += 1

            if built == un:
                break
            if not un.startswith(built) and not built.startswith(un):
                break

        if acc is None:
            # fallback consume one token
            acc = np.array(vals[ti], dtype=float)
            ti += 1

        out.append({"Word": u, "SHAP_HUMAN": float(acc[0]), "SHAP_BOT": float(acc[1])})

        if ti == start_ti:
            ti += 1

    dfw = pd.DataFrame(out)
    if dfw.empty:
        return pd.DataFrame(columns=["Word","SHAP_HUMAN","SHAP_BOT"])

    # aggregate duplicates (surface word may appear multiple times in text)
    dfw = dfw.groupby("Word", as_index=False)[["SHAP_HUMAN","SHAP_BOT"]].sum()
    return dfw

# -------------------- Global Aggregation: mean(|SHAP_class|) --------------------
def aggregate_global_mean_abs(explanations, texts):
    """
    explanations: shap.Explanation for texts (len N), each has exp.data[i], exp.values[i]
    returns two dicts:
      mean_abs_human[token] = mean abs SHAP_HUMAN across examples where token appears
      mean_abs_bot[token]   = mean abs SHAP_BOT   across examples where token appears
    """
    sum_h, cnt_h = {}, {}
    sum_b, cnt_b = {}, {}

    for i in tqdm(range(len(texts)), desc="Aggregate"):
        text = texts[i]
        exp_i = explanations[i]

        shap_tokens = list(exp_i.data)            # tokens for this example
        shap_vals   = exp_i.values                # (n_tokens, 2)

        # optional truncation to keep consistent with SHAP_MAX_LEN
        # (Explainer already truncates via masker tokenizer, but we also truncate projection input length)
        if shap_vals.ndim != 2 or shap_vals.shape[1] != 2:
            continue

        # project to surface "words"
        dfw = project_token_shap_to_surface(text, shap_tokens, shap_vals)
        if dfw.empty:
            continue

        # mean(|SHAP_class|) is computed across examples where unit appears
        for _, r in dfw.iterrows():
            w = str(r["Word"])
            vh = abs(float(r["SHAP_HUMAN"]))
            vb = abs(float(r["SHAP_BOT"]))
            sum_h[w] = sum_h.get(w, 0.0) + vh
            cnt_h[w] = cnt_h.get(w, 0) + 1
            sum_b[w] = sum_b.get(w, 0.0) + vb
            cnt_b[w] = cnt_b.get(w, 0) + 1

    mean_abs_h = {w: sum_h[w] / max(cnt_h[w], 1) for w in sum_h}
    mean_abs_b = {w: sum_b[w] / max(cnt_b[w], 1) for w in sum_b}
    return mean_abs_h, mean_abs_b

def to_rank_df(mean_dict, colname="mean_|SHAP|"):
    rows = sorted(mean_dict.items(), key=lambda x: x[1], reverse=True)
    return pd.DataFrame(rows, columns=["Token", colname])

def take_topk_with_backfill(rank_df: pd.DataFrame, ban: set, k: int) -> pd.DataFrame:
    sel = []
    for _, r in rank_df.iterrows():
        tok = r["Token"]
        if tok in ban:
            continue
        sel.append(r)
        if len(sel) == k:
            break
    return pd.DataFrame(sel, columns=rank_df.columns)

def make_exclusive(rank_bot: pd.DataFrame, rank_hum: pd.DataFrame, k: int):
    bot_sel = take_topk_with_backfill(rank_bot, ban=set(), k=k)
    ban = set(bot_sel["Token"])
    hum_sel = take_topk_with_backfill(rank_hum, ban=ban, k=k)
    return bot_sel.reset_index(drop=True), hum_sel.reset_index(drop=True)

# -------------------- Compute SHAP --------------------

t0 = time.time()
# explainer(texts, ...) returns shap.Explanation with len = N
shap_vals_global = explainer(texts_global, max_evals=MAX_EVALS, batch_size=BATCH_SIZE)
print(f"[Timing] SHAP computed in {time.time()-t0:.1f}s")

# Optional: um Projektion stabil zu halten, kürzen wir die Tokens über den Tokenizer selbst
# (nicht zwingend; wenn du willst, kann man hier vorher texts_global tokenisieren und kürzen)


# -------------------- Global Aggregation --------------------
mean_abs_h, mean_abs_b = aggregate_global_mean_abs(shap_vals_global, texts_global)

rank_human = to_rank_df(mean_abs_h, colname="mean_|SHAP_HUMAN|")
rank_bot   = to_rank_df(mean_abs_b, colname="mean_|SHAP_BOT|")

if EXCLUSIVE_TOPK:
    bot_top, human_top = make_exclusive(rank_bot, rank_human, k=TOP_K)
else:
    bot_top   = rank_bot.head(TOP_K).reset_index(drop=True)
    human_top = rank_human.head(TOP_K).reset_index(drop=True)

print("\n=== SHAP GLOBAL – Top Tokens (mean |SHAP|) – BOT output ===")
try: display(bot_top)
except Exception: print(bot_top.to_string(index=False))

print("\n=== SHAP GLOBAL – Top Tokens (mean |SHAP|) – HUMAN output ===")
try: display(human_top)
except Exception: print(human_top.to_string(index=False))
    

# -------------------- Export author_ids --------------------
pd.DataFrame({"author_id": author_ids_global}).to_csv(EXPORT_ORDERED_IDS_CSV, index=False)
print(f"\n[Export] Ordered author IDs -> {EXPORT_ORDERED_IDS_CSV}  (n={len(author_ids_global)})")

unique_ids = sorted(set(author_ids_global))
pd.DataFrame({"author_id": unique_ids}).to_csv(EXPORT_UNIQUE_IDS_CSV, index=False)
print(f"[Export] Unique author IDs   -> {EXPORT_UNIQUE_IDS_CSV}  (n={len(unique_ids)})")

sample_df_out = sample_df.copy()
sample_df_out.insert(0, "row_idx", np.arange(len(sample_df_out)))
sample_df_out.to_csv(EXPORT_SAMPLE_ROWS_CSV, index=False)
print(f"[Export] Sample rows -> {EXPORT_SAMPLE_ROWS_CSV}")


