# Explainable AI Methods for Understanding Decisions of Text-based Language Models

This repository contains the code accompanying a Master’s thesis on  
text-based bot detection using transformer models, with a focus on  
model interpretability through SHAP.

The project investigates how explainable AI methods can be applied to  
analyze and better understand the decisions of fine-tuned language  
models in a bot-vs-human classification setting.

---

## Repository Structure

The repository is organized into the following main components:

**src/**  
- **preprocessing/** Data preparation, filtering, sampling, statistics  
- **training/**   Model training, author-disjoint splits, evaluation  
- **explainability/** SHAP-based local and global explainability analyses  

**plots/**  
- Scripts for generating visualizations  


**data/**  
- `example_data.csv` Synthetic placeholder illustrating the data schema  

---

## Data Availability

The original Twitter dataset used in this project is **not included**  
in the repository due to licensing and privacy restrictions.

The file `data/example_data.csv` contains **synthetic placeholder data**  
illustrating the expected data format required by the codebase.  
It does **not** contain real tweets or user information.

---

## Model Training and Evaluation

Model training and evaluation are implemented in the `src/training/`  
directory. The code includes:

- fine-tuning of transformer-based language models  
- author-disjoint data splitting  
- evaluation using standard classification metrics  
- comparison of different data split configurations  

---

## Explainability

Explainability analyses are implemented in `src/explainability/`  
and are based on SHAP.

- **Global SHAP** Analysis of aggregated feature contributions  
- **Local SHAP** Qualitative case analysis of individual predictions  

Local SHAP results are presented as **structured example analyses**  
rather than graphical SHAP plots.

---

## Visualizations 

Scripts for generating figures and tables used in the thesis are  
located in the `plots/` directory.

---

## Reproducibility Notes

- The repository focuses on **code-level reproducibility**  
- Generated artefacts (model checkpoints, outputs, intermediate files)  
  are created locally and are not versioned  
- Model training was conducted in a cloud-based environment  
  (Google Colab) using an NVIDIA A100 GPU  

---

## Context

This repository accompanies a Master’s thesis in the field of  
digital media, natural language processing, and explainable  
artificial intelligence.
