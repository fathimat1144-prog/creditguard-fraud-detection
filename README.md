# 🔍 CreditGuard — Credit Card Fraud Detection System

> Detect fraudulent transactions with high accuracy using Isolation Forest anomaly detection and XGBoost classification, powered by SMOTE oversampling to handle extreme class imbalance.

---

## 📌 Project Overview

Financial fraud is a critical real-world problem with severe class imbalance — only **0.17%** of transactions are fraudulent. CreditGuard tackles this with a multi-model pipeline that combines unsupervised anomaly detection and supervised classification, with careful handling of the imbalance problem and rigorous evaluation using industry-standard metrics.

| Property | Value |
|---|---|
| Dataset | [Kaggle Credit Card Fraud Detection](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud) |
| Rows | 284,807 transactions |
| Fraud Cases | 492 (0.17%) |
| Features | 30 (V1–V28 PCA-anonymized + Amount + Time) |
| Best AUC-ROC | **0.9809** |

---

## 🧠 Models Used

| Model | Type | Notes |
|---|---|---|
| **Isolation Forest** | Anomaly Detection | Unsupervised; no labels needed at inference |
| **XGBoost + SMOTE** | Classification | Best performer; AUC-ROC = 0.98 |
| **Random Forest + SMOTE** | Classification | Baseline comparison with balanced class weights |

---

## ⚙️ Techniques

- **SMOTE** (Synthetic Minority Over-sampling Technique) — balances training data from 578:1 to 1:1
- **RobustScaler** — scales 'Amount' and 'Time' features; resilient to outliers
- **Threshold Optimization** — sweeps 0.01–0.99 to find the F1-maximizing decision boundary
- **Stratified Train/Test Split** — preserves fraud rate in both splits

---

## 📊 Results

| Model | F1-Score | AUC-ROC | Avg Precision |
|---|---|---|---|
| Isolation Forest | 0.2705 | — | — |
| Random Forest + SMOTE | ~0.85 | ~0.97 | ~0.78 |
| XGBoost + SMOTE (threshold=0.5) | 0.4211 | 0.9809 | — |
| XGBoost + SMOTE (tuned threshold) | **Best F1** | **0.9809** | High |

> ⚠️ Isolation Forest operates without labels, so lower F1 is expected. Its AUC-ROC is computed from anomaly scores.

---

## 📁 Project Structure

```
creditguard-fraud-detection/
│
├── fraud_detection.ipynb     # Main notebook (all models + visualizations)
├── creditcard.csv            # Dataset (download from Kaggle — not included)
├── requirements.txt          # Python dependencies
└── README.md                 # This file
```

---

## 🚀 Getting Started

### 1. Clone the Repository

```bash
git clone https://github.com/your-username/creditguard-fraud-detection.git
cd creditguard-fraud-detection
```

### 2. Install Dependencies

```bash
pip install -r requirements.txt
```

### 3. Download the Dataset

Download 'creditcard.csv' from [Kaggle](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud) and place it in the project root directory.

### 4. Run the Notebook

```bash
jupyter notebook fraud_detection.ipynb
```

Run all cells top to bottom. Total runtime: ~5–10 minutes depending on hardware.

---

## 📦 Requirements

See ['requirements.txt'](requirements.txt) for the full list. Key dependencies:

- 'scikit-learn' — models, metrics, preprocessing
- 'xgboost' — gradient boosted classifier
- 'imbalanced-learn' — SMOTE oversampling
- 'pandas' / `numpy' — data manipulation
- 'matplotlib' / 'seaborn' — visualization

---

## 📈 Visualizations Generated

The notebook produces the following plots (saved as '.png'):

| File | Description |
|---|---|
| 'eda_overview.png' | Class distribution, amount & time histograms |
| 'feature_analysis.png' | Top discriminating features + correlation heatmap |
| 'feature_boxplots.png' | Box plots for top 6 fraud features |
| 'smote_balance.png' | Before/after SMOTE class balance |
| 'confusion_matrices.png' | Side-by-side confusion matrices for all models |
| 'roc_pr_curves.png' | ROC + Precision-Recall curves |
| 'feature_importance.png' | XGBoost top 20 feature importances |
| 'threshold_optimization.png' | F1/Precision/Recall vs. decision threshold |
| 'score_distributions.png' | Anomaly score & probability distributions |
| 'model_comparison.png' | Bar chart comparing all metrics across models |

---

## 🔑 Key Findings

- **V14, V17, V12, V10, V16** are the top fraud-discriminating PCA features
- SMOTE on a 578:1 imbalance dramatically improves recall without much precision loss
- XGBoost achieves AUC-ROC of **0.9809** — near-perfect class separation
- Isolation Forest is valuable when labeled data is unavailable; it still ranks most fraud cases highly anomalous
- Default threshold of 0.5 is suboptimal for imbalanced data — tuning it significantly improves F1

---

## 🧪 Evaluation Metrics Explained

- **F1-Score** — harmonic mean of Precision and Recall; ideal for imbalanced classes
- **AUC-ROC** — probability that the model ranks a fraud case above a legitimate one
- **Average Precision (AP)** — area under the Precision-Recall curve; better than AUC-ROC when positives are rare
- **Confusion Matrix** — breakdown of TP, FP, FN, TN for operational insight

---

## 📚 Dataset

The dataset is the [Credit Card Fraud Detection](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud) dataset from Kaggle, provided by the Machine Learning Group at ULB (Université Libre de Bruxelles).

- Features V1–V28 are the result of PCA transformation (original features are confidential)
- 'Time' = seconds elapsed from first transaction in the dataset
- 'Amount' = transaction amount in euros
- 'Class' = 1 (fraud) / 0 (legitimate)

---

## 🤝 Contributing

Pull requests are welcome. For major changes, please open an issue first to discuss what you'd like to change.

1. Fork the repository
2. Create your branch ('git checkout -b feature/your-feature')
3. Commit your changes ('git commit -m 'Add your feature')
4. Push to the branch ('git push origin feature/your-feature')
5. Open a Pull Request

---

## 📄 License

This project is open-source and available under the [MIT License](LICENSE).

---

## 🙏 Acknowledgements

- Dataset: [MLG-ULB](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud) via Kaggle
- [imbalanced-learn](https://imbalanced-learn.org/) for SMOTE implementation
- [XGBoost](https://xgboost.readthedocs.io/) team
