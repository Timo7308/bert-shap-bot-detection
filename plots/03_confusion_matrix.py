import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap

cm = np.array([
    [4591, 3792],  # True Human: TN, FP
    [2510, 6503]   # True Bot:   FN, TP
])

labels = ["Human", "Bot"]

cm_percent = cm / cm.sum(axis=1, keepdims=True)

annot = np.empty_like(cm).astype(str)
for i in range(cm.shape[0]):
    for j in range(cm.shape[1]):
        count = cm[i, j]
        perc = cm_percent[i, j] * 100
        annot[i, j] = f"{count}\n({perc:.1f}%)"

# Farbverlauf: Rot → Weiß → Grün
colors = ["#ff4c4c", "#ffffff", "#4caf50"]
cmap = LinearSegmentedColormap.from_list("custom_cmap", colors, N=256)

# Plot
plt.figure(figsize=(7,6))
sns.heatmap(cm_percent,
            annot=annot,
            fmt='',
            cmap=cmap,
            cbar=False,
            linewidths=0.3,
            linecolor='black',
            square=True,
            xticklabels=[f"Pred: {l}" for l in labels],
            yticklabels=[f"True: {l}" for l in labels],
            annot_kws={"size": 13})


plt.xlabel("Predicted class", fontsize=12)
plt.ylabel("Actual class", fontsize=12)
plt.yticks(rotation=0)
plt.xticks(rotation=0)
plt.tight_layout()
plt.show()
