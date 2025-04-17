# pcos-classification-comparative-studies

## Description

This repository provides a comparative analysis between **Scikit-learn ensemble classifiers** and **custom-built classification algorithms** implemented from scratch. All models are trained and evaluated on the same dataset to ensure a fair and consistent performance comparison.

## Experiments

The experiments are conducted on a real-world medical dataset related to PCOS (Polycystic Ovary Syndrome), sourced from Kaggle.

We compare several hybrid classification strategies that combine data resampling techniques with ensemble classifiers. In addition to standard combinations like SMOTE + Random Forest, we evaluate two recent methods proposed in the literature: **SWSEL** and **VASA**.

### Highlights:
- **Feature selection**: Performed using Mutual Information
- **Sampling strategy**: Class balance ratio set to 0.75 (minority/majority)
- **Validation**: Stratified 5-Fold Cross-Validation
- **Evaluation metrics**: Precision, Recall, F1-score, G-mean
- **Hyperparameter tuning**: Grid search over number of base classifiers, learning rate, and method-specific parameters (e.g., alpha for VASA)
