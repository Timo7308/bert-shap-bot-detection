# Explainable AI Methods for Understanding Decisions of Text-based Language Models

This repository contains the code accompanying a Master’s thesis on
text-based bot detection using transformer models with a focus on
model interpretability through SHAP.

The project investigates how explainable AI methods can be used to
analyze and better understand the decisions of fine-tuned language
models in a bot-vs-human classification setting.

---

## Repository Structure

The repository is organized into the following main components:

src/  
- preprocessing/  Data preparation, filtering, sampling, statistics  
- training/    Model training, author-disjoint splits, evaluation  
- explainability/ SHAP-based local and global explainability analyses  

plots/  
- Scripts for result visualizations  

reports/  
- Figures and tables used in the written thesis  

data/  
- example_data.csv Synthetic placeholder illustrating the data schema  
- README.md    Notes on data availability and structure  

---

## Data Availability

The original Twitter dataset used in this project is **not included**
in this repository due to licensing and privacy restrictions.

The file `data/example_data.csv` contains **synthetic placeholder data**
that illustrates the expected data format required by the codebase.
It does **not** contain real tweets or user information.

---

## Model Training and Evaluation

Model training and evaluation are implemented in the `src/training/`
directory. The code includes:

- fine-tuning of transformer-based language models  
- author-disjoint data splitting  
- evaluation using standard classification metrics  
- comparison of different data split configurations  

All training-related logic is separated from visualization and
explainability code.

---

## Explainability

Explainability analyses are implemented in `src/explainability/` and
are based on SHAP.

- **Global SHAP** is used to analyze aggregated feature contributions
  across multiple samples.  
- **Local SHAP** is used for qualitative case analysis of individual
  predictions.  

Local SHAP results are presented as **structured example analyses**
rather than graphical SHAP plots.

---

## Visualizations and Reports

Scripts for generating figures and tables used in the thesis are
located in the `plots/` directory.

Final figures and tables intended for inclusion in the written thesis
are collected under `reports/`.

---

## Reproducibility Notes

- The repository focuses on **code-level reproducibility**.  
- Generated artefacts (model checkpoints, outputs, intermediate files)
  are created locally and are not versioned.  
- Model training was conducted in a cloud-based environment
  (Google Colab) using an NVIDIA A100 GPU.

## Context

This repository accompanies a Master’s thesis in the field of
digital media, natural language processing, and explainable
artificial intelligence.

