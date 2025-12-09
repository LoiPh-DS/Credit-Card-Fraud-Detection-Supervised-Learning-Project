# Credit Card Fraud Detection: Machine Learning Project

**Author:** Loi Pham
**Date:** December 2025

## Project Overview

This project implements and compares multiple supervised learning algorithms for detecting fraudulent credit card transactions in a highly imbalanced dataset.

## Problem Statement

Credit card fraud detection with:
- 50,000 transactions
- Only 2.49% fraud rate (1,246 fraudulent transactions)
- 39:1 class imbalance
- Goal: Detect fraud while minimizing false positives

## Methods

Implemented and compared four supervised learning models:
1. **Logistic Regression** - Baseline linear model
2. **Support Vector Machine (SVM)** - RBF kernel for non-linear patterns
3. **Random Forest** - Ensemble method
4. **AdaBoost** - Sequential ensemble

All models used `class_weight='balanced'` to handle class imbalance.

## Results

**Best Model: Support Vector Machine (SVM)**

| Metric | Value |
|--------|-------|
| Accuracy | 97.97% |
| Precision | 57.23% |
| Recall | 73.09% |
| F1-Score | 64.20% |
| ROC-AUC | 91.37% |

**Key Performance:**
- Catches 182 out of 249 frauds (73% recall)
- Only 136 false positives out of 9,751 legitimate transactions
- Estimated $1M annual savings after review costs

## Technologies Used

- Python 3.8+
- scikit-learn (machine learning)
- pandas, numpy (data processing)
- matplotlib, seaborn (visualization)
- Jupyter Notebook

## How to Run

1. Clone this repository
2. Install dependencies: `pip install numpy pandas scikit-learn matplotlib seaborn jupyter`
3. Open `fraud_detection_project.ipynb` in Jupyter
4. Run all cells

## Dataset

Credit Card Fraud Detection dataset from Kaggle (or synthetic data generated in notebook).
- 30 features (V1-V28 PCA-transformed, Time, Amount)
- Binary classification (0=legitimate, 1=fraud)

## Key Findings

- SVM achieved optimal balance between precision and recall
- Logistic Regression had too many false positives
- Random Forest and AdaBoost were too conservative (missed 98% of frauds)
- Feature V19, V14, and V1 were most important for detection

## Future Improvements

- Implement SMOTE oversampling
- Try deep learning approaches
- Add real-time prediction API
- Hyperparameter tuning with GridSearchCV

## License

This project is for educational purposes.
