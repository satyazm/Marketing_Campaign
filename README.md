# Marketing Campaign Classification

A machine learning project that predicts customer conversion in digital marketing campaigns and compares classification algorithms end-to-end — from EDA through hyperparameter-tuned, calibrated, stacked models.

![Python](https://img.shields.io/badge/Python-3.7%2B-blue)
![Jupyter](https://img.shields.io/badge/Jupyter-Notebook-orange)
![scikit--learn](https://img.shields.io/badge/scikit--learn-ML-f7931e)
![License](https://img.shields.io/badge/License-Unspecified-lightgrey)

<details>
<summary><strong>📑 Table of Contents</strong> (click to expand)</summary>

- [Project Overview](#-project-overview)
- [Objective](#-objective)
- [Dataset](#-dataset)
- [Notebook Structure](#-notebook-structure)
- [Machine Learning Models](#-machine-learning-models)
- [Technologies & Libraries](#️-technologies--libraries)
- [Installation](#-installation)
- [Usage](#-usage)
- [Model Evaluation Metrics](#-model-evaluation-metrics)
- [Results Snapshot](#-results-snapshot)
- [Key Insights](#-key-insights)
- [Acknowledgments](#-acknowledgments)

</details>

## 📋 Project Overview

This project analyzes a digital marketing campaign dataset to predict whether a customer will convert (make a purchase) based on demographic, behavioral, and campaign-related features. It covers exploratory data analysis (EDA), feature engineering, model building, hyperparameter tuning, calibration, and ensemble stacking.

The notebook, [MARKETING-CAMPAIGN-CLASSIFICATION.ipynb](MARKETING-CAMPAIGN-CLASSIFICATION.ipynb), is organized into six numbered top-level sections (see [Notebook Structure](#-notebook-structure) below). Each section header collapses in Jupyter/VS Code, and supplementary commentary is tucked into `<details>` dropdowns so the core analysis stays easy to scan.

## 🎯 Objective

Build a robust binary classification model that can:
- Predict customer conversion with high accuracy
- Identify key factors influencing conversion rates
- Optimize marketing campaign strategies
- Calculate important business metrics (CPA, ROI, CTR)

## 📊 Dataset

The project uses `digital_marketing_campaign_dataset.csv`, which is **not included in this repository** — place it in the project root before running the notebook.

**Features include:**
- **Customer Demographics**: Age, Gender, Income
- **Campaign Details**: CampaignChannel, CampaignType, AdSpend
- **Behavioral Metrics**: ClickThroughRate, WebsiteVisits, PagesPerVisit, TimeOnSite
- **Social Metrics**: SocialShares, EmailOpens, EmailClicks
- **Target Variable**: Conversion (Binary: 0 or 1)

**Dataset characteristics:**
| Property | Value |
|---|---|
| Total Records | 8,000 rows |
| Total Features | 20 columns (18 after dropping confidential columns) |
| Class Distribution | Imbalanced (handled via SMOTE) |
| Missing / Duplicate Values | None |

## 🗂️ Notebook Structure

<details>
<summary><strong>Full section breakdown</strong> (click to expand)</summary>

**1. Exploring the Dataframe**
- 1.1 Categorical Variables Selection Criteria
- 1.2 What We Find Out About the Dataframe
- 1.3 Further Observations About the Data

**2. Campaign Success Metrics**
- 2.1 Conversion Rate
- 2.2 Cost per Acquisition (CPA)
- 2.3 Return on Investment (ROI)
- 2.4 Click-Through Rate (CTR)
- 2.5 Bounce Rate
- 2.6 Average Transaction Value
- 2.7 Conversion Rate by Campaign Channel

**3. Demography Analysis**
- 3.1 Customer Demographics
- 3.2 Income Segmentation
- 3.3 Channel-Specific Demographics

**4. Customer Retention Metrics**
- 4.1 Returning Visitors
- 4.2 Email Engagement
- 4.3 Previous Purchases

**5. Cost Related Charts**
- 5.1 Ad Spend vs. Conversion
- 5.2 Cost Breakdown by Channel

**6. Predictive Modelling**
- 6.1 Data Preprocessing
- 6.2 Comprehensive Model Building Plan
  - Step 1 — Experiment Header & Settings
  - Step 2 — Splitting Strategy & Hold-out Lock
  - Step 3 — Preprocessing Blueprint (Fold-Aware)
  - Step 4 — Class Imbalance Strategy (P0)
  - Step 5 — Baselines: Dummy & Regularized Logistic
  - Step 6 — Model Roster & Uniform Evaluation Loop
  - Step 7-8 — Hyperparameter Optimization (HPO)
  - Step 9 — Calibration Pass (P0 Requirement)
  - Step 10 — Threshold Selection (Operating Points)
  - Step 11 — OOF Stacking (P0 Requirement)

</details>

## 🤖 Machine Learning Models

The project implements and compares:

1. Logistic Regression (regularized baseline)
2. Random Forest Classifier
3. XGBoost Classifier
4. LightGBM Classifier
5. CatBoost Classifier
6. Support Vector Machine (SVM)
7. K-Nearest Neighbors (KNN)
8. Multi-Layer Perceptron (MLP) Classifier
9. Gradient Boosting Classifier
10. Decision Tree Classifier

**Advanced techniques:**
- Leak-safe, fold-aware preprocessing and stratified K-fold cross-validation
- Class imbalance handling (SMOTE / SMOTETomek)
- Randomized hyperparameter search under a time budget
- Probability calibration (Platt / isotonic) with reliability analysis
- Threshold tuning for business-aligned operating points
- Out-of-fold (OOF) stacking ensemble

## 🛠️ Technologies & Libraries

| Category | Libraries |
|---|---|
| Core | pandas, numpy, scipy |
| Visualization | matplotlib, seaborn |
| Machine Learning | scikit-learn, xgboost, lightgbm, catboost |
| Imbalanced Data | imbalanced-learn (SMOTE) |
| Utilities | pickle, tqdm |

## 📦 Installation

**Prerequisites:** Python 3.7+ and Jupyter Notebook

```bash
pip install numpy pandas matplotlib seaborn scikit-learn
pip install xgboost lightgbm catboost
pip install imbalanced-learn scipy tqdm
```

## 🚀 Usage

1. **Clone the repository**
   ```bash
   git clone https://github.com/satyazm/Marketing_Campaign.git
   cd Marketing_Campaign
   ```

2. **Add the dataset**
   Place `digital_marketing_campaign_dataset.csv` in the project root.

3. **Open the notebook**
   ```bash
   jupyter notebook MARKETING-CAMPAIGN-CLASSIFICATION.ipynb
   ```

4. **Run cells sequentially** — the Table of Contents dropdown at the top of the notebook links to each section. Cells are numbered to match: EDA (§1) → campaign/demographic/cost analytics (§2-5) → preprocessing and the Step 1-11 model-building pipeline (§6).

## 📈 Model Evaluation Metrics

Models are evaluated using:
- **Accuracy** — overall prediction correctness
- **Precision** / **Recall** / **F1-Score**
- **PR-AUC** and **ROC-AUC**
- **Brier Score** and **Log Loss** — probability calibration quality
- **Confusion Matrix** — classification performance visualization
- **Cross-Validation Scores** — model stability across folds

## 🏆 Results Snapshot

From the notebook's model comparison (Step 6 evaluation, before HPO/calibration/stacking):

| Model | PR-AUC | Accuracy | Notes |
|---|---|---|---|
| CatBoost | 0.9566 | 92.38% | Best overall |
| Random Forest | 0.9546 | 88.86% | Well-balanced |
| Gradient Boosting | 0.9542 | 92.02% | Low overfitting |

Subsequent steps (7-11) apply hyperparameter search, probability calibration, threshold selection, and OOF stacking on top of these baselines — see the notebook for final numbers.

## 🔑 Key Insights

- Feature importance rankings across candidate models
- Demographic analysis (age, income, gender) vs. conversion behavior
- Channel effectiveness comparison (Email, Social Media, PPC, SEO, Influencer)
- Cost-benefit analysis (CPA, ROI, ad spend vs. conversion)
- Customer retention signals (returning visitors, email engagement, repeat purchases)
- Full model performance comparison, calibration quality, and stacked ensemble lift

## 🙏 Acknowledgments

- Dataset source: Digital Marketing Campaign Dataset
- Inspiration: real-world marketing analytics challenges
- Tools: Kaggle environment used for initial development

---

**Note**: This project demonstrates an end-to-end machine learning workflow for business analytics, from data exploration to model deployment readiness.
