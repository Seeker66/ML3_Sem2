# Intrusion Detection using Machine Learning

This repository contains a generalized approach for evaluating **machine learning models** on **network intrusion detection data**.  
The project investigates how different models and feature selection strategies can best categorize and detect malicious activity from network traffic captures.

---

## Project Overview
- **Goal:** Detect & classify malicious traffic using supervised ML  
- **Dataset:** Network flow-based features (initially 79 columns → cleaned to 38)  
- **Problem type:** Multiclass classification  
- **Performance Metrics**:  
  * **Accuracy**  
  * **Weighted Precision** (accounts for class imbalance by weighting each class by frequency)  
  * **Macro Precision** (treats all classes equally, important for minority classes)  
  * **Confusion Matrices**  

---

## Dataset Preparation

### 1. Cleaning & Preprocessing
- Dropped NaN and Inf values  
- Removed near-constant features (Tolerance of `>0.95`)
- Reduced highly correlated features (Tolerance of `>0.95`)
- Dataset reduced from **79 → 38 features (+1 target)**

### 2. Splitting
- **Training:** 70% (831,833 rows)  
- **Validation:** 15% (178,250 rows)  
- **Testing:** 15% (178,250 rows)  

---
## Feature Selection & Dimensionality Reduction

### **Pipeline A: PCA (95% variance explained)**
- Features reduced: **38 → 19 components**  
- Explained variance ≈ **99.9998%**  
- R² = 0.61 (moderate reconstruction)  
- ![PCA Variance Plot](images/pca_variance.png)

---

### **Pipeline B: Random Forest + Permutation Importance**
- Features reduced: **38 → 21 features**  
- Most important features:  
  - `Destination Port`  
  - `Fwd IAT Min`, `Flow IAT Min`, `Fwd IAT Std`  
- Dropped uninformative (p=1.0) features  
- ![Feature Importance](images/feature_importance.png)

---

### **Pipeline C: Random Forest + PCA**
- Features reduced: **21 → 12 components**  
- Explained variance ≈ **99.99999%**  
- R² = 0.68 (better than PCA alone)  

---

## Models & Training

Implemented via a generalized `run_classifiers()` function:
- Trains Ridge, Logistic Regression, Decision Tree, and KNN  
- Supports **class weighting** for imbalanced datasets  
- Saves models & encoders for reuse  
- Generates:
  - Accuracy scores  
  - Weighted & Macro Precision  
  - Confusion matrices  

---

## Results


### **Pipeline A – PCA only (19 components)**

| Model                | Accuracy | Weighted Precision | Macro Precision | Notes |
|----------------------|----------|-------------------|----------------|-------|
| Logistic Regression   | 98.7%    | 0.987             | ~0.6–0.7       | Struggles on minority classes |
| Decision Tree         | 99.98%   | 0.999             | 0.985          | Strongest performer |
| KNN                   | 99.97%   | 0.999             | 0.983          | Nearly as strong |
| Ridge Classifier      | ~98–99%  | High              | Low            | Unstable |

| Accuracy Comparison | Weighted vs Macro |
|--------------------|-----------------|
| ![Pipeline A Accuracy Comparison](https://raw.githubusercontent.com/Seeker66/ML3_Sem2/main/graphics/pipelineA_accuracy_comparison.png) | ![Pipeline A Weighted vs Macro](https://raw.githubusercontent.com/Seeker66/ML3_Sem2/main/graphics/pipelineA_weighted_vs_macro.png) |

### Confusion Matrices

<table>
<tr>
  <td><img src="https://raw.githubusercontent.com/Seeker66/ML3_Sem2/main/graphics/pipelineA_dtree_matrix.png" width="200"></td>
  <td><img src="https://raw.githubusercontent.com/Seeker66/ML3_Sem2/main/graphics/pipelineA_knn_matrix.png" width="200"></td>
  <td><img src="https://raw.githubusercontent.com/Seeker66/ML3_Sem2/main/graphics/pipelineA_unweighted_log_matrix.png" width="200"></td>
</tr>
<tr>
  <td><img src="https://raw.githubusercontent.com/Seeker66/ML3_Sem2/main/graphics/pipelineA_unweighted_ridge_matrix.png" width="200"></td>
  <td><img src="https://raw.githubusercontent.com/Seeker66/ML3_Sem2/main/graphics/pipelineA_weighted_log_matrix.png" width="200"></td>
  <td><img src="https://raw.githubusercontent.com/Seeker66/ML3_Sem2/main/graphics/pipelineA_weighted_ridge_matrix.png" width="200"></td>
</tr>
</table>

---

### **Pipeline B – Random Forest (21 selected features)**

| Model                | Accuracy | Weighted Precision | Macro Precision | Notes |
|----------------------|----------|-------------------|----------------|-------|
| Logistic Regression   | 64–92%   | 0.64–0.92         | Very low       | Fails to detect minority classes |
| Decision Tree         | 99.98%   | 0.998             | 0.981          | Very strong |
| KNN                   | 99.96%   | 0.997             | 0.959          | Small macro precision drop |
| Ridge Classifier      | ~92%     | Decent            | Weak           | Weaker than tree/KNN |


| Accuracy Comparison | Weighted vs Macro |
|--------------------|-----------------|
| ![Pipeline B Accuracy Comparison](https://raw.githubusercontent.com/Seeker66/ML3_Sem2/main/graphics/pipelineB_accuracy_comparison.png) | ![Pipeline B Weighted vs Macro](https://raw.githubusercontent.com/Seeker66/ML3_Sem2/main/graphics/pipelineB_weighted_vs_macro.png) |

### Confusion Matrices

<table>
<tr>
  <td><img src="https://raw.githubusercontent.com/Seeker66/ML3_Sem2/main/graphics/pipelineB_dtree_matrix.png" width="200"></td>
  <td><img src="https://raw.githubusercontent.com/Seeker66/ML3_Sem2/main/graphics/pipelineB_knn_matrix.png" width="200"></td>
  <td><img src="https://raw.githubusercontent.com/Seeker66/ML3_Sem2/main/graphics/pipelineB_unweighted_log_matrix.png" width="200"></td>
</tr>
<tr>
  <td><img src="https://raw.githubusercontent.com/Seeker66/ML3_Sem2/main/graphics/pipelineB_unweighted_ridge_matrix.png" width="200"></td>
  <td><img src="https://raw.githubusercontent.com/Seeker66/ML3_Sem2/main/graphics/pipelineB_weighted_log_matrix.png" width="200"></td>
  <td><img src="https://raw.githubusercontent.com/Seeker66/ML3_Sem2/main/graphics/pipelineB_weighted_ridge_matrix.png" width="200"></td>
</tr>
</table>

---

### **Pipeline C – Hybrid (RF → PCA, 12 components)**

| Model                | Accuracy | Weighted Precision | Macro Precision | Notes |
|----------------------|----------|-------------------|----------------|-------|
| Logistic Regression   | 98.1%    | 0.981             | Poor           | Consistently weak on minorities |
| Decision Tree         | 99.97%   | 0.997             | 0.961          | Strong, slightly lower macro |
| KNN                   | 99.97%   | 0.998             | 0.980          | Strong all-round |
| Ridge Classifier      | ~98%     | Good              | Weak           | Similar weakness trend |

| Accuracy Comparison | Weighted vs Macro |
|--------------------|-----------------|
| ![Pipeline C Accuracy Comparison](https://raw.githubusercontent.com/Seeker66/ML3_Sem2/main/graphics/pipelineC_accuracy_comparison.png) | ![Pipeline C Weighted vs Macro](https://raw.githubusercontent.com/Seeker66/ML3_Sem2/main/graphics/pipelineC_weighted_vs_macro.png) |

### Confusion Matrices

<table>
<tr>
  <td><img src="https://raw.githubusercontent.com/Seeker66/ML3_Sem2/main/graphics/pipelineC_dtree_matrix.png" width="200"></td>
  <td><img src="https://raw.githubusercontent.com/Seeker66/ML3_Sem2/main/graphics/pipelineC_knn_matrix.png" width="200"></td>
  <td><img src="https://raw.githubusercontent.com/Seeker66/ML3_Sem2/main/graphics/pipelineC_unweighted_log_matrix.png" width="200"></td>
</tr>
<tr>
  <td><img src="https://raw.githubusercontent.com/Seeker66/ML3_Sem2/main/graphics/pipelineC_unweighted_ridge_matrix.png" width="200"></td>
  <td><img src="https://raw.githubusercontent.com/Seeker66/ML3_Sem2/main/graphics/pipelineC_weighted_log_matrix.png" width="200"></td>
  <td><img src="https://raw.githubusercontent.com/Seeker66/ML3_Sem2/main/graphics/pipelineC_weighted_ridge_matrix.png" width="200"></td>
</tr>
</table>


---

## Key Findings

### Top 10 Models Across All Pipelines

| Rank | Model | Pipeline | Accuracy | Weighted Precision | Macro Precision |
|------|-------|---------|---------|------------------|----------------|
| 1    | Decision Tree | B | 0.9998 | 0.9998 | 0.9880 |
| 2    | Decision Tree | A | 0.9998 | 0.9998 | 0.9853 |
| 3    | KNN           | A | 0.9997 | 0.9997 | 0.9828 |
| 4    | KNN           | C | 0.9997 | 0.9997 | 0.9803 |
| 5    | Decision Tree | C | 0.9997 | 0.9997 | 0.9614 |
| 6    | Unweighted Logistic Regression | A | 0.9867 | 0.9817 | 0.3940 |
| 7    | Unweighted Logistic Regression | C | 0.9812 | 0.9765 | 0.3929 |
| 8    | Weighted Logistic Regression   | A | 0.9616 | 0.9863 | 0.4651 |
| 9    | L2 Ridge (SGD)                 | A | 0.9472 | 0.9785 | 0.4214 |
| 10   | Unweighted Ridge               | B | 0.9217 | 0.9219 | 0.3766 |

### Pipeline-Wide Performance Comparison (Average of All Models)

| Pipeline | Avg Accuracy | Avg Weighted Precision | Avg Macro Precision |
|----------|-------------|----------------------|-------------------|
| A        | 0.9427      | 0.9705               | 0.6581            |
| B        | 0.8737      | 0.9374               | 0.4936            |
| C        | 0.9163      | 0.9604               | 0.5483            |

**Summary:**

- Pipeline A shows the strongest overall performance, both in accuracy and macro precision.  
- Pipeline B contains the lowest performing models in terms of consistency and macro precision.  
- Pipeline C is competitive with high scores, but slightly behind Pipeline A in consistency.

---

## How to Use

### 1. Data Preprocessing
- Load dataset CSV into a Pandas DataFrame.  
- Apply cleaning:
  - Remove NaN and Inf values.  
  - Drop low-variance and highly correlated features.  
- Output: cleaned dataset with ~64 columns.  

### 2. Feature Reduction
Choose one of the three pipelines:
- **PCA**: Reduce dimensions while retaining variance.  
- **Random Forest**: Select most important features. Permutation analysis and shifting is conducted to likewise calculate significance of features.
- **RF + PCA**: Combine feature importance with dimensionality reduction.  


### 3. Running Classifiers
Use the provided function:
```python
run_classifiers(X_train, y_train, X_val, y_val, label_column)


