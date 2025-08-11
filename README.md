# Fraud Detection Project – August 2025

**Team:** Sasha, Dana, Murat  
**Best Zindi Score:** 0.083 (Rank 193/595)  
**Date:** August 2025  

---

## 📌 Summary

This project focuses on detecting fraudulent transactions in an **imbalanced dataset**, where the ratio of class 0 (non-fraud) to class 1 (fraud) is heavily skewed.  
We explored multiple approaches to handle class imbalance and improve model performance, starting from a baseline model to advanced feature engineering with XGBoost.  
Our best public Zindi leaderboard score was **0.083**, placing us **193rd out of 595** teams.

---

## 🔄 Pipeline Overview

### **a) Baseline Model – Logistic Regression**
- Implemented as an initial benchmark.
- Used default settings without balancing adjustments.
- Highlighted the impact of class imbalance: good accuracy for class 0, poor recall for class 1.
- Served as a performance baseline for later improvements.

### **b) Testing Samplers & XGBoost with `scale_pos_weight`**
- Implemented in **`W3_SMOTE_SMOTETOMEK_ADA_XGBOOST_WITH_EXPORT.ipynb`**.
- Tested **imbalanced-learn samplers**:
  - **SMOTE** – Synthetic Minority Over-sampling Technique.
  - **ADASYN** – Adaptive Synthetic Sampling.
  - **SMOTETomek** – Combination of over-sampling and cleaning via Tomek links.
- Compared with **XGBoost using `scale_pos_weight`**:
  - Uses the class ratio `(negatives / positives)` directly inside the model.
  - No synthetic data is generated; weighting is applied in the loss function.
- **Key difference:**  
  - **Samplers** adjust the dataset.  
  - **`scale_pos_weight`** adjusts how the model learns from the unbalanced data.

### **c) XGBoost + `scale_pos_weight` with Feature Engineering**
- Implemented in **`W4_XGBOOST_scale_pos_weight_wtih_ranking_and_feature_combinations.ipynb`**.
- Added **feature engineering** to improve discriminatory power.
- Used **RandomizedSearchCV** for hyperparameter tuning.
- Evaluated with **StratifiedKFold** to preserve class ratio in each fold.
- Measured metrics with **cross_validate** for a consistent and fair evaluation.
- Determined optimal probability thresholds using **precision_recall_curve** to maximize F1 score.

---

## ⚙️ Methodological Choices

### **RandomizedSearchCV**
- Efficient hyperparameter search over a defined parameter distribution.
- Used `f1_score` as the primary scoring metric to focus on balancing precision and recall.

### **StratifiedKFold**
- Ensures the **class 0 / class 1 ratio** is preserved in each fold.
- Prevents folds without minority-class examples, which would distort evaluation.

### **cross_validate**
- Used after model selection to evaluate multiple metrics (accuracy, balanced accuracy, precision, recall, F1, ROC-AUC, etc.) on the same folds.
- Provided a consistent performance profile without changing the chosen model.

### **precision_recall_curve**
- Explored the trade-off between precision and recall at different thresholds.
- Selected thresholds that maximize F1 score, especially important in fraud detection where recall is critical.

---

## 📦 Why `imblearn.pipeline` Was a Good Choice
- Allowed direct integration of **sampling techniques** into the modeling pipeline.
- Ensured **resampling happens inside each CV fold**, avoiding data leakage from the test folds.
- Enabled easy comparison between different samplers and the `scale_pos_weight` approach.

---

## 🔍 Samplers vs. `scale_pos_weight`

| Approach            | How it works | Pros | Cons |
|---------------------|--------------|------|------|
| **SMOTE**           | Synthesizes new minority-class examples. | Improves class balance in training; may improve recall. | Risk of overfitting; adds artificial data. |
| **ADASYN**          | Like SMOTE, but focuses on harder-to-learn minority samples. | Targets difficult cases. | May introduce noisy samples. |
| **SMOTETomek**      | Combines SMOTE with Tomek links to remove borderline/noisy points. | Cleans dataset while balancing. | More complex, can remove useful borderline data. |
| **`scale_pos_weight`** | Adjusts loss function weight using class ratio (neg/pos). | No new data; efficient; works well with tree models like XGBoost. | May underperform if feature space is sparse or imbalanced within subgroups. |

---

## 📊 Class Imbalance Context
- Original dataset had a **severe imbalance** between class 0 and class 1.
- **StratifiedKFold** preserved this ratio in validation.
- **Samplers** and **`scale_pos_weight`** addressed it in different ways:
  - Samplers: physically adjust the dataset distribution.
  - `scale_pos_weight`: adjusts learning inside the model based on the class ratio.

---

## 📅 Project Workflow
1. **EDA & Baseline Logistic Regression** (`W1_EDA_FEATURE_BASELINE_MUTUAL.ipynb`, `W1_EDA_FEATURE_BASELINE_LR_MK.ipynb`)
2. **Sampler & XGBoost Experiments** (`W3_SMOTE_SMOTETOMEK_ADA_XGBOOST_WITH_EXPORT.ipynb`)
3. **Feature Engineering & Optimized XGBoost** (`W4_XGBOOST_scale_pos_weight_wtih_ranking_and_feature_combinations.ipynb`)
4. **Export Predictions** and submit to Zindi leaderboard.
5. Achieved **0.083** public leaderboard score (**Rank 193/595**).

---

# `fraud.py` – Class & Function Reference

This section documents the **`Fraud`** class and the helper functions provided in `fraud.py`.  
It includes **usage examples**, a full **API reference**, and clearly marks which functions are **aggregation-based feature generators**.

> ℹ️ Note: Aggregation functions compute statistics over groups (e.g., per `client_id`, `region`, `district`) and are the core of our feature engineering for imbalanced fraud detection.

---

## 1) Overview

The `Fraud` class is a lightweight loader/organizer for multiple CSV sources (client, invoice, …).  
It offers dict-like access to loaded `pandas` DataFrames and provides helpers to **merge** datasets and **assemble features**.

Key ideas:

- Preserve **class ratio** with `StratifiedKFold` in validation.
- Create **aggregation features** (per client/region/district).
- Avoid **leakage**: target-dependent aggregations must be computed **within** CV folds only.

---

## 2) Quick Start Example

```python
from fraud import (
    Fraud, 
    left_join_on,
    add_invoice_frequency_features,
    add_counter_statue_error_occured_features,
    add_region_fraud_rate_features  # target-based (⚠ leakage risk)
)

# 1) Load sources (+ optional target extraction)
fraud = Fraud(
    csv_files=[
        "./data/train/client_train.csv",
        "./data/train/invoice_train.csv"
    ],
    target_column="target"
)

# 2) Access sources
client  = fraud["./data/train/client_train.csv"]
invoice = fraud["./data/train/invoice_train.csv"]

# 3) Build a merged base for feature engineering
merged = left_join_on("client_id", client, invoice)

# 4) Start feature frame with identity/geo + (optional) target
features = fraud.get_target(client)

# 5) Add aggregation features
features = add_invoice_frequency_features(merged, features)
features = add_counter_statue_error_occured_features(merged, features)

# 6) (Train-time only, inside CV fold) add target-based aggregates
features = add_region_fraud_rate_features(merged, features)

print(features.head())
```
## 3) Minimal CV Sketch

```
from imblearn.pipeline import Pipeline
from xgboost import XGBClassifier
from sklearn.model_selection import StratifiedKFold, RandomizedSearchCV
from sklearn.metrics import make_scorer, f1_score

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
f1_scorer = make_scorer(f1_score, pos_label=1, zero_division=0)

pipeline = Pipeline([
    # ("sampler", SMOTE(random_state=42)),  # optional: data-level balancing
    ("xgb", XGBClassifier(eval_metric="logloss", random_state=42))
])

param_dist = {
    "xgb__n_estimators": randint(100, 500),
    "xgb__max_depth": randint(3, 10),
    "xgb__learning_rate": uniform(0.01, 0.3),
    "xgb__subsample": uniform(0.5, 0.5),
    "xgb__colsample_bytree": uniform(0.5, 0.5),
    "xgb__min_child_weight": randint(1, 10),
    "xgb__gamma": uniform(0, 5),
    # "xgb__scale_pos_weight": [ratio],    # model-level balancing
}

search = RandomizedSearchCV(
    estimator=pipeline,
    param_distributions=param_dist,
    n_iter=30,
    scoring=f1_scorer,
    cv=cv,
    n_jobs=-1,
    random_state=42
)

search.fit(X_train, y_train)
best_model = search.best_estimator_
```

--- 
# 📘 Feature Engineering Functions Overview

## 9) Feature Glossary (Selected)

| Feature name | Type | Description |
|--------------|------|-------------|
| `f_invoive_date_diff_days` | aggregation | Median days between invoices per client. |
| `f_invoive_date_median_months` | aggregation | Median months between invoices per client. |
| `f_invoive_date_median_years` | aggregation | Median years between invoices per client. |
| `f_counter_statue_error_occured` | aggregation | 1 if any non-zero `counter_statue` occurred; else 0. |
| `f_counter_regions` | aggregation | 1 if client has invoices across >1 regions; else 0. |
| `f_t_region_fraud_rate` | aggregation (target-based) | Region-level mean fraud rate mapped to clients. |
| `f_region_median_billing_frequence_per` | aggregation | Regional median of client-level invoice spacing. |
| `f_region_std_deviation_consumption_*` | aggregation | Regional std of specific consumption level. |
| `f_index_diff_*` | aggregation | Stats of `new_index - old_index` per client (min/max/mean/std). |
| `f_total_consumption_*` | aggregation | Stats of total consumption per client. |
| `f_tarif_type_mode` | aggregation | Most common tariff type per client. |
| `f_t_district_target_mean` | aggregation (target-based) | District-level mean fraud rate. |
| `f_t_client_catg_target_mean` | aggregation (target-based) | Client-category-level mean fraud rate. |
| `f_index_cons_error_sum` | aggregation | Sum of `(new-old) - total_consumption`. |
| `f_counter_statue_mean` | aggregation | Mean of encoded `counter_statue` per client. |
| `f_client_tenure_days` | aggregation | Days between creation and last invoice. |
| `f_counter_number_nunique` | aggregation | Unique meters observed per client. |
| `f_tarif_change_count` | aggregation | Count of distinct tariff types per client. |
| `avg_consumption_per_month` | aggregation | Average monthly consumption per client. |
| `remarque_frequency` | aggregation | Ratio of invoices with `reading_remarque`. |
| `avg_remarque_length` | aggregation | Average length of `reading_remarque`. |

---
## Set up your Environment



### **`macOS`** type the following commands : 



- For installing the virtual environment and the required package you can either follow the commands:

    ```BASH
    pyenv local 3.11.3
    python -m venv .venv
    source .venv/bin/activate
    pip install --upgrade pip
    pip install -r requirements.txt
    ```

### **`WindowsOS`** type the following commands :

- Install the virtual environment and the required packages by following commands.

   For `PowerShell` CLI :

    ```PowerShell
    pyenv local 3.11.3
    python -m venv .venv
    .venv\Scripts\Activate.ps1
    pip install --upgrade pip
    pip install -r requirements.txt
    ```

    For `Git-bash` CLI :
  
    ```BASH
    pyenv local 3.11.3
    python -m venv .venv
    source .venv/Scripts/activate
    pip install --upgrade pip
    pip install -r requirements.txt
    ```

    **`Note:`**
    If you encounter an error when trying to run `pip install --upgrade pip`, try using the following command:
    ```Bash
    python.exe -m pip install --upgrade pip
    ```


