# Marketing Campaign Classification

A comprehensive machine learning project for predicting customer conversion in digital marketing campaigns using various classification algorithms and advanced techniques.

## 📋 Project Overview

This project analyzes a digital marketing campaign dataset to predict whether a customer will convert (make a purchase) based on various demographic, behavioral, and campaign-related features. The analysis includes exploratory data analysis (EDA), feature engineering, model building, hyperparameter tuning, and ensemble methods.

## 🎯 Objective

The main goal is to build a robust binary classification model that can:
- Predict customer conversion with high accuracy
- Identify key factors influencing conversion rates
- Optimize marketing campaign strategies
- Calculate important business metrics (CPA, ROI, CTR)

## 📊 Dataset

The project uses the `digital_marketing_campaign_dataset.csv` which contains customer and campaign information:

### Features Include:
- **Customer Demographics**: Age, Gender, Income
- **Campaign Details**: CampaignChannel, CampaignType, AdSpend
- **Behavioral Metrics**: ClickThroughRate, WebsiteVisits, PagesPerVisit, TimeOnSite
- **Social Metrics**: SocialShares, EmailOpens, EmailClicks
- **Target Variable**: Conversion (Binary: 0 or 1)

### Dataset Characteristics:
- **Total Records**: 8,000 rows
- **Total Features**: 20 columns (18 after dropping confidential columns)
- **Class Distribution**: Imbalanced dataset (handled using SMOTE)

## 🔍 Key Analyses

### 1. Campaign Success Metrics
- **Conversion Rate**: Percentage of users who converted
- **Cost per Acquisition (CPA)**: Total campaign cost divided by acquired customers
- **Return on Investment (ROI)**: Revenue comparison to campaign cost
- **Click-Through Rate (CTR)**: Percentage of clicks relative to impressions
- **Bounce Rate**: Percentage of single-page visits

### 2. Data Preprocessing
- Handling outliers and skewed distributions
- Power transformations for skewed variables
- One-hot encoding for categorical variables (CampaignChannel, CampaignType, Gender)
- Feature scaling and normalization

### 3. Class Imbalance Handling
- Implementation of SMOTE (Synthetic Minority Over-sampling Technique)
- Stratified train-test splitting

## 🤖 Machine Learning Models

The project implements and compares multiple classification algorithms:

1. **Logistic Regression** (Baseline with regularization)
2. **Random Forest Classifier**
3. **XGBoost Classifier**
4. **LightGBM Classifier**
5. **CatBoost Classifier**
6. **Support Vector Machine (SVM)**
7. **K-Nearest Neighbors (KNN)**
8. **Multi-Layer Perceptron (MLP) Classifier**
9. **Gradient Boosting Classifier**
10. **Decision Tree Classifier**

### Advanced Techniques:
- **Hyperparameter Optimization**: Extensive grid/random search
- **Cross-Validation**: Stratified K-Fold validation
- **Model Calibration**: Probability calibration for better predictions
- **Ensemble Methods**: Out-of-Fold (OOF) stacking
- **Threshold Tuning**: Operating point selection

## 🛠️ Technologies & Libraries

**Core Libraries:**
- pandas
- numpy
- scipy

**Visualization:**
- matplotlib
- seaborn

**Machine Learning:**
- scikit-learn
- xgboost
- lightgbm
- catboost

**Imbalanced Data:**
- imblearn (SMOTE)

**Utilities:**
- pickle
- tqdm

## 📦 Installation

### Prerequisites
```bash
Python 3.7+
Jupyter Notebook
```

### Install Dependencies
```bash
pip install numpy pandas matplotlib seaborn scikit-learn
pip install xgboost lightgbm catboost
pip install imbalanced-learn scipy tqdm
```

## 🚀 Usage

1. **Clone the repository**:
```bash
git clone https://github.com/ashwini0008/MARKETING-CAMPAIGN-CLASSIFICATION.git
cd MARKETING-CAMPAIGN-CLASSIFICATION
```

2. **Ensure the dataset is available**:
- Place `digital_marketing_campaign_dataset.csv` in the root directory

3. **Open and run the Jupyter notebook**:
```bash
jupyter notebook MARKETING-CAMPAIGN-CLASSIFICATION.ipynb
```

4. **Execute cells sequentially** to:
   - Load and explore the data
   - Perform EDA and visualizations
   - Preprocess and transform features
   - Train multiple models
   - Evaluate and compare results
   - Generate predictions

## 📈 Model Evaluation Metrics

The models are evaluated using:
- **Accuracy**: Overall prediction correctness
- **Precision**: Positive prediction accuracy
- **Recall**: Actual positive identification rate
- **F1-Score**: Harmonic mean of precision and recall
- **ROC-AUC**: Area under the ROC curve
- **Confusion Matrix**: Classification performance visualization
- **Cross-Validation Scores**: Model stability assessment

## 🔑 Key Insights

The notebook provides comprehensive analysis including:
- Feature importance rankings
- Demographic analysis (age, income, gender)
- Channel effectiveness comparison
- Cost-benefit analysis
- Customer retention metrics
- Model performance comparisons




##  Acknowledgments

- Dataset source: Digital Marketing Campaign Dataset
- Inspiration: Real-world marketing analytics challenges
- Tools: Kaggle environment for initial development

---

**Note**: This project demonstrates end-to-end machine learning workflow for business analytics, from data exploration to model deployment readiness.
# Marketing_Campaign
