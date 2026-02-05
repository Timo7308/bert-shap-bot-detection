# bertweet-shap-bot-detection

This repository contains the code accompanying a Master’s thesis on explainable
text-based bot detection using transformer models and SHAP.

---

## 1. Project Overview

- What is the central research objective of this project?
- Which classification task is addressed (e.g. bot vs. human on Twitter)?
- Which model architecture is used as the primary focus, and why?
- Why is explainability a core aspect beyond predictive performance?
- In which academic context was this project developed?

---

## 2. Dataset

- Which dataset is used as the data source?
- At which level are labels defined (author-level vs. tweet-level)?
- Why are the original data files not included in this repository?
- Which file illustrates the expected data schema?
- Where are dataset-specific details documented?

(See `data/README.md` for dataset-related notes.)

---

## 3. Repository Structure

The repository is structured into the following main components:

- `src/preprocessing/`  
  – Scripts for data filtering, deduplication, sampling, and descriptive statistics.

- `src/training/`  
  – Model training, author-disjoint data splitting, evaluation, and configuration.

- `src/explainability/`  
  – SHAP-based local and global explainability analyses.

- `plots/`  
  – Scripts for generating figures used in the results section of the thesis.

- `reports/`  
  – Folder intended for final figures and tables used in the written thesis.

---

## 4. Preprocessing

- What is the initial format of the raw data?
- Which preprocessing steps are applied before model training?
- How is the final working sample constructed?
- Which design decisions are applied to ensure robustness and reproducibility?
- Which scripts implement these steps?

---

## 5. Training and Evaluation

- Which model is fine-tuned for the classification task?
- How are training, validation, and test splits defined?
- Why are author-disjoint splits used?
- Which evaluation metrics are reported?
- How is early stopping and model selection handled?

---

## 6. Explainability

- Which explainability method is applied?
- What is the distinction between local and global explainability in this project?
- At which level are explanations computed (token-level, aggregated units)?
- Which approximations or assumptions are made?
- Where is the explainability logic implemented?

---

## 7. Visualizations and Reports

- Which results are visualized using plots?
- Which aspects of model performance are compared graphically?
- Why are no classical SHAP local plots included?
- How are local SHAP results presented instead?
- Which outputs are intended for the final thesis document?

---

## 8. Reproducibility Notes

- Which Python version and core libraries are required?
- Are there hardware assumptions (e.g. GPU availability)?
- Which artefacts are generated locally but not versioned?
- What is required to conceptually reproduce the experiments?

---

## 9. Context

This repository accompanies a Master’s thesis in the field of digital media,
machine learning, and explainable artificial intelligence.
