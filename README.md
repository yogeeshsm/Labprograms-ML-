
# 🧠 Machine Learning Lab Programs – VTU Curriculum

This repository contains implementations of core machine learning algorithms and analysis techniques as per VTU syllabus using datasets like California Housing, Iris, Breast Cancer, Olivetti Faces, etc. All programs are implemented in **Python** using libraries such as `NumPy`, `Pandas`, `Matplotlib`, `Seaborn`, `Scikit-learn`.

---

## 📁 Folder Structure

```
machine-learning-lab/
│
├── 01_histogram_boxplot_california.py
├── 02_correlation_heatmap_pairplot.py
├── 03_pca_iris.py
├── 04_find_s_algorithm.py
├── 05_knn_classification_random_values.py
├── 06_locally_weighted_regression.py
├── 07_linear_polynomial_regression.py
├── 08_decision_tree_breast_cancer.py
├── 09_naive_bayes_olivetti_faces.py
├── 10_kmeans_wisconsin_cancer.py
├── datasets/
│   ├── california_housing.csv
│   ├── iris.csv
│   ├── boston.csv
│   ├── auto_mpg.csv
│   ├── breast_cancer.csv
│   └── ...
└── README.md
```

---

## 🔧 Requirements

Install the required libraries using:

```bash
pip install numpy pandas matplotlib seaborn scikit-learn opencv-python
```

---

## 📜 Lab Programs

### 📊 Program 1

**Title:** Create histograms & box plots for all numerical features.
**Dataset:** California Housing
**Objective:** Analyze distributions and detect outliers.
**Textbook 1:** Chapter 2

---

### 🔥 Program 2

**Title:** Compute and visualize correlation matrix and pair plots.
**Dataset:** California Housing
**Objective:** Analyze inter-feature relationships.
**Textbook 1:** Chapter 2

---

### 📉 Program 3

**Title:** Dimensionality Reduction using PCA
**Dataset:** Iris
**Objective:** Reduce 4D feature space to 2D using PCA.
**Textbook 1:** Chapter 2

---

### 📐 Program 4

**Title:** Implement Find-S algorithm
**Dataset:** Custom (CSV format)
**Objective:** Output hypothesis space from training data.
**Textbook 1:** Chapter 3

---

### 🤖 Program 5

**Title:** KNN Classification of randomly generated data
**Dataset:** Synthetic (x ∈ \[0,1])
**Objective:** Classify using k-NN for multiple values of k.
**Textbook 2:** Chapter 2

---

### 📈 Program 6

**Title:** Locally Weighted Regression (LWR)
**Dataset:** Custom/Synthetic
**Objective:** Implement non-parametric LWR and visualize fit.
**Textbook 1:** Chapter 4

---

### 📊 Program 7

**Title:** Linear & Polynomial Regression
**Dataset:** Boston Housing (Linear), Auto MPG (Polynomial)
**Objective:** Compare linear vs non-linear regression performance.
**Textbook 1:** Chapter 5

---

### 🌲 Program 8

**Title:** Decision Tree Classifier
**Dataset:** Breast Cancer Dataset
**Objective:** Build decision tree and classify new sample.
**Textbook 2:** Chapter 3

---

### 🧠 Program 9

**Title:** Naive Bayes Classifier
**Dataset:** Olivetti Face Dataset
**Objective:** Implement classifier and calculate accuracy.
**Textbook 2:** Chapter 4

---

### 📍 Program 10

**Title:** K-Means Clustering
**Dataset:** Wisconsin Breast Cancer Dataset
**Objective:** Perform clustering and visualize results.
**Textbook 2:** Chapter 4

---

## ✅ Usage

To run any program:

```bash
python program_name.py
```

Ensure the required dataset is present in the `datasets/` directory.

