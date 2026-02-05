import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import pandas as pd
import seaborn as sns

# =============================
# Performance of BERT across all epochs
# =============================

epochs = [1, 2, 3, 4]
val_f1 = [0.6221, 0.6290, 0.6288, 0.6271]
val_loss = [0.6450, 0.6389, 0.6626, 0.6947]

fig = plt.figure(figsize=(8, 3.5))
gs = GridSpec(1, 2, wspace=0.45) 

# Links: F1 (grün)
ax1 = fig.add_subplot(gs[0, 0])
ax1.plot(epochs, val_f1, marker='o', color='green')
ax1.set_xlabel("Epoch")
ax1.set_ylabel("Validation F1")
ax1.set_ylim(0.60, 0.65)
ax1.set_title("")

# Rechts: Loss (rot)
ax2 = fig.add_subplot(gs[0, 1])
ax2.plot(epochs, val_loss, marker='s', color='red')
ax2.set_xlabel("Epoch")
ax2.set_ylabel("Validation Loss")
ax2.set_title("")

plt.show()

# =============================
# Comparison of dataset sizes and splits
# =============================

sizes = [2000, 5000, 10000, 20000, 30000, 40000, 50000, 60000]

# Test-F1-Werte
test_701515 = [0.5954, 0.6039, 0.6175, 0.6267, 0.6179, 0.6176, 0.6278, 0.6327]
test_801010 = [0.5686, 0.6100, 0.6162, 0.6308, 0.6281, 0.6236, 0.6355, 0.6360]

# Gemeinsame Skala berechnen
all_test = test_801010 + test_701515
ymin = round(min(all_test) - 0.01, 2)
ymax = round(max(all_test) + 0.01, 2)

# Plot
fig, ax = plt.subplots(figsize=(9,5))

ax.plot(sizes, test_801010, marker="o", linestyle="-", label="Test F1 (80-10-10)", color="tab:green")
ax.plot(sizes, test_701515, marker="s", linestyle="-", label="Test F1 (70-15-15)", color="tab:red")

# Formatierung

ax.set_xlabel("Trainingsgröße", fontsize=12)
ax.set_ylabel("Macro-F1", fontsize=12)
ax.set_xticks(sizes)
ax.set_xticklabels([f"{s//1000}k" for s in sizes])
ax.set_ylim(ymin, ymax)
ax.grid(alpha=0.3)
ax.legend(loc="lower right", fontsize=10)
plt.tight_layout()
plt.show()



# =============================
# Comparison of different BERT variants
# =============================


data = {
    "model": [
        "bertweet-base",
        "NeoBERT",
        "deberta-base",
        "roberta-base",
        "MiniLM-L12"
    ],
    "f1_macro": [0.630792, 0.622046, 0.617366, 0.617176, 0.616414],
    "f1_human": [0.610730, 0.629286, 0.632148, 0.608432, 0.576817],
    "f1_bot": [0.650855, 0.614806, 0.602584, 0.625921, 0.656012]
}

df = pd.DataFrame(data)

# Nach F1 Macro sortieren
df_macro_sorted = df.sort_values(by='f1_macro', ascending=False)

plt.figure(figsize=(8, 5))
sns.barplot(data=df_macro_sorted, x='model', y='f1_macro', color='steelblue')
plt.title("")
plt.ylabel("F1 Macro")
plt.ylim(0.60, 0.635)
plt.xticks(rotation=45, ha='right')

ax = plt.gca()
for p in ax.patches:
    ax.text(p.get_x() + p.get_width()/2, p.get_height() + 0.001, f"{p.get_height():.3f}",
            ha='center', va='bottom', fontsize=9)

plt.tight_layout()
plt.show()
