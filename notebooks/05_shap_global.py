import pandas as pd

bot_data = {
    "Token": ["glover","thirty","accidentally","alt","faze5","auditory","night's","benefitted","wi","glam"],
    "mean_shap": [0.302986,0.230945,0.229539,0.182994,0.174292,0.162645,0.158493,0.152410,0.152171,0.150093]
}

human_data = {
    "Token": ["setback","becaus","cello","semi","critic","confirms","mzala","personally","neuromarketing","sparked"],
    "mean_shap": [0.124668,0.123128,0.122026,0.119501,0.116601,0.115720,0.114698,0.111068,0.111021,0.108854]
}

bot_top10 = pd.DataFrame(bot_data)
human_top10 = pd.DataFrame(human_data)

import matplotlib.pyplot as plt

bot_color   = "#c0392b"
human_color = "#27ae60"

fig, axes = plt.subplots(2, 1, figsize=(8, 9), sharex=True)

# BOT
axes[0].barh(bot_top10["Token"], bot_top10["mean_shap"], color=bot_color)
axes[0].set_title("Global SHAP – BOT", pad=10)
axes[0].set_xlabel("mean |SHAP|")   # <- jetzt auch oben
axes[0].tick_params(labelbottom=True)
axes[0].invert_yaxis()

# HUMAN
axes[1].barh(human_top10["Token"], human_top10["mean_shap"], color=human_color)
axes[1].set_title("Global SHAP – HUMAN", pad=10)
axes[1].set_xlabel("mean |SHAP|")
axes[1].invert_yaxis()

for ax in axes:
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

plt.subplots_adjust(hspace=0.45)
plt.show()
