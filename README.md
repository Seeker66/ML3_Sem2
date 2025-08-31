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

### Pipeline Results

| Pipeline | Model               | Accuracy | Weighted Precision | Macro Precision | Notes |
|----------|---------------------|----------|--------------------|-----------------|-------|
| **Pipeline A - PCA only (19 components)** | Decision Tree       | 99.98%   | 0.999             | 0.985           | Strongest performer |
|          | KNN                  | 99.97%   | 0.999             | 0.983           | Nearly as good |
|          | Logistic Regression  | 98.70%   | 0.987             | Poor (~0.6–0.7) | Weak for minority classes |
|          | Ridge Classifier     | ~98–99%  | High              | Low             | Unstable |
| **Pipeline B - Random Forest with 50 permutations (21 features)** | Decision Tree       | 99.98%   | 0.998             | 0.981           | Very strong |
|          | KNN                  | 99.96%   | 0.997             | 0.959           | Small macro drop |
|          | Logistic Regression  | 64–92%   | 0.64–0.92         | Very low        | Fails minority detection |
|          | Ridge Classifier     | ~92%     | Decent            | Weak            | Poorer than tree/KNN |
| **Pipeline C - Hybrid (RF → PCA, 12 comps)** | Decision Tree       | 99.97%   | 0.997             | 0.961           | Strong, slightly lower macro |
|          | KNN                  | 99.97%   | 0.998             | 0.980           | Strong all-round |
|          | Logistic Regression  | 98.10%   | 0.981             | Poor            | Consistent weakness |
|          | Ridge Classifier     | ~98%     | Good              | Weak            | Same trend |


### **Pipeline A (PCA only)**
- Logistic Regression (unweighted): **98.7% acc**  
- Decision Tree: **99.98% acc, Macro ≈ 0.985**  
- KNN: **99.97% acc, Macro ≈ 0.983**  
- ![Confusion Matrix PCA](images/confusion_matrix_pca.png)

---

### **Pipeline B (RF Features)**
- Logistic Regression: **64–92% acc**, poor Macro precision  
- Decision Tree: **99.98% acc, Macro ≈ 0.981**  
- KNN: **99.96% acc, Macro ≈ 0.959**  
- ![Confusion Matrix RF](images/confusion_matrix_rf.png)

---

### **Pipeline C (RF + PCA)**
- Logistic Regression (unweighted): **98.1% acc**  
- Decision Tree: **99.97% acc, Macro ≈ 0.961**  
- KNN: **99.97% acc, Macro ≈ 0.980**  
- ![Confusion Matrix RF+PCA](images/confusion_matrix_rfpca.png)

---

## Key Findings
- **Decision Tree consistently best performer** across all pipelines  
- **Feature reduction improves training speed & accuracy** vs using raw dataset  
- **KNN nearly matches Decision Tree** performance  
- **Logistic Regression & Ridge** underperform on minority classes (poor Macro scores)  
- **Pipeline B (RF feature selection)** gave best balance of **interpretability & accuracy**

---

## Visual Outputs
- PCA variance explained curve  
- Feature importance plot  
- Accuracy comparison chart  
- Weighted vs Macro Precision comparison  
- Confusion matrices  

(Placeholders above — embed your figures later)

---
**Summary:**  
- **Decision Tree** and **KNN** dominate across pipelines.  
- PCA provided excellent dimensionality reduction with minimal loss of information.  
- Logistic Regression underperformed due to sensitivity to imbalance.  
- Random Forest feature selection validated that ~21 features carry most predictive power.  

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
- **Random Forest**: Select most important features.  
- **RF + PCA**: Combine feature importance with dimensionality reduction.  

### 3. Running Classifiers
Use the provided function:
```python
run_classifiers(X_train, y_train, X_val, y_val, label_column)
