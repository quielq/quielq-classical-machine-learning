# Machine Learning Projects: AI 221 & AI 201

Coursework projects in classical machine learning: supervised classification and regression, dimensionality reduction, clustering, and anomaly detection. Everything is built with scikit-learn in Jupyter notebooks.

Each project was originally its own repository. They're consolidated here and the original commit history is preserved (`git log -- ai221/<project>`).

| Course | Focus |
|---|---|
| **AI 221: Classical Machine Learning** | Seven machine exercises and projects covering linear models, kernel methods, ensembles, neural networks, manifold learning, and unsupervised learning |
| **AI 201: Artificial Intelligence** | Mini-project comparing four classifiers on a nutrition dataset, written up as an ACM-style report |

---

## Repository map

```
.
├── ai221/
│   ├── energy-efficiency-in-buildings/        ME 1  · Ridge regression, regularization, feature importance
│   ├── palmer-penguin-species-classifier/     ME 2  · RBF SVM, decision boundaries
│   ├── wine-quality/                          ME 3  · SVM, MLP, Random Forest, Gradient Boosting (classification + regression)
│   ├── handwritten-digits/                    ME 4  · PCA, Kernel PCA, LDA, t-SNE, Isomap, LLE, Laplacian Eigenmaps
│   ├── country-development-classifier/        ME 5  · K-means, hierarchical clustering, PCA, Isolation Forest
│   ├── breast-tumor-classifier/               Logistic regression, ROC/AUC, coefficient interpretation
│   └── laguna-lake-chlorophyll-A-prediction/  Gaussian Process Regression, Kernel Ridge Regression
└── ai201/
    └── foodclassifier/                        Naive Bayes, SVM, MLP, AdaBoost + ANOVA feature selection
```

### Quick navigation

| # | Project | Task | Main notebook | Best result |
|---|---|---|---|---|
| ME 1 | [Energy Efficiency in Buildings](#1-energy-efficiency-in-buildings) | Regression | [`energyefficiency.ipynb`](ai221/energy-efficiency-in-buildings/energyefficiency.ipynb) | R² > 0.90 with Ridge (α = 1e-4) |
| ME 2 | [Palmer Penguin Species](#2-palmer-penguin-species-classifier) | Multiclass classification | [`palmerpenguinspecies.ipynb`](ai221/palmer-penguin-species-classifier/palmerpenguinspecies.ipynb) | 98.8% test accuracy, macro-F1 0.986 |
| ME 3 | [Wine Quality](#3-wine-quality) | Classification + regression | [`winequality.ipynb`](ai221/wine-quality/winequality.ipynb) | 68.75% acc. / MAD 0.41 (Gradient Boosting) |
| ME 4 | [Handwritten Digits](#4-handwritten-digits) | Dim. reduction + classification | [`digits.ipynb`](ai221/handwritten-digits/digits.ipynb) | 99.17% acc. (KPCA + SVC) |
| ME 5 | [Country Development](#5-country-development-classifier) | Clustering + anomaly detection | [`countrydata.ipynb`](ai221/country-development-classifier/countrydata.ipynb) | 5 clusters (Ward linkage) |
| — | [Breast Tumor Classifier](#6-breast-tumor-classifier) | Binary classification | [`classifying_breast_tumors.ipynb`](ai221/breast-tumor-classifier/classifying_breast_tumors.ipynb) | Test AUC 0.9975, F1 0.96 |
| — | [Laguna Lake Chlorophyll-A](#7-laguna-lake-chlorophyll-a-prediction) | Regression | [`chlorophyllprediction.ipynb`](ai221/laguna-lake-chlorophyll-A-prediction/chlorophyllprediction.ipynb) | Test R² 0.86 (GPR) |
| AI 201 | [Food Classifier](#8-food-classifier-for-fitness-goals-ai-201) | Multiclass classification | [`ai201/foodclassifier/`](ai201/foodclassifier) | F1 0.983 (AdaBoost) |

---

## AI 221: Classical Machine Learning

### 1. Energy Efficiency in Buildings
📁 [`ai221/energy-efficiency-in-buildings`](ai221/energy-efficiency-in-buildings) · *with Joel Paolo C. Salvador*

Predicts the **heating load** of 768 simulated buildings from 8 architectural attributes: compactness, surface/wall/roof area, height, orientation, and glazing.

- **Approach:** `StandardScaler → Ridge` pipeline with a train/validation/test split. Ten regularization strengths (α = 1e-4 … 1000) are compared on validation data, then the best is scored once on the test set.
- **AI 221 concepts:**
  - **Linear regression with L2 regularization (Ridge)** and the bias–variance tradeoff. Large α (100, 1000) visibly underfits.
  - **Model selection with a held-out validation set**, kept separate from the final test evaluation.
  - **Interpreting standardized coefficients** as feature importance (sign = direction, magnitude = strength).
  - **Feature selection:** retraining on the top 5 features (height, compactness, roof area, surface area, glazing area) shows almost no loss in R².
- **Data:** `ENB2012_data.xlsx` (UCI Energy Efficiency)

### 2. Palmer Penguin Species Classifier
📁 [`ai221/palmer-penguin-species-classifier`](ai221/palmer-penguin-species-classifier) · *with Joel Paolo C. Salvador*

Classifies Adélie, Chinstrap, and Gentoo penguins using only **culmen length** and **flipper length**.

- **Approach:** Pairplot EDA, then a stratified 75/25 split and a `StandardScaler → SVC(kernel='rbf')` pipeline. Decision regions are plotted with `DecisionBoundaryDisplay`.
- **AI 221 concepts:**
  - **Support Vector Machines** and the **kernel trick** (RBF) for non-linear decision boundaries.
  - **Why feature scaling matters** for distance- and margin-based models.
  - **Visualizing decision boundaries** in a 2-D feature space.
  - **Multiclass metrics:** confusion matrix, accuracy, macro-F1.
- **Result:** 94.5% train / 98.8% test accuracy. Chinstrap, which overlaps the other two species, is the hardest class.

### 3. Wine Quality
📁 [`ai221/wine-quality`](ai221/wine-quality)

Predicts expert quality scores for 1,599 red *Vinho Verde* wines from 11 physicochemical attributes. The goal is to improve on the benchmark paper (Cortez et al., SVR with MAD 0.46).

- **Approach:** Four model families are tuned with `GridSearchCV` (5-fold CV) and each is trained **as a classifier and as a regressor**: SVM/SVR, MLP, Random Forest, and Gradient Boosting.
- **AI 221 concepts:**
  - **Kernel SVM / SVR**, **Multilayer Perceptrons** (architecture, activation, L2 penalty), **bagging (Random Forest)**, and **boosting (Gradient Boosting)**.
  - **Hyperparameter tuning with cross-validation** (`GridSearchCV`, `RandomizedSearchCV`).
  - **Class imbalance:** the rare quality scores 3, 4, and 8 are never predicted, and the notebook discusses the effect of a stratified split.
  - **Classification vs. regression for ordinal targets:** regression respects the ordering of scores, while classification treats them as unordered labels.
  - **Regression metrics:** MSE, R², MAD.
- **Result:** Gradient Boosting was best in both settings: **68.75% accuracy** (paper: 62.4%) and **MAD 0.410** (paper: 0.46).

### 4. Handwritten Digits
📁 [`ai221/handwritten-digits`](ai221/handwritten-digits)

Visualizes and classifies scikit-learn's 8×8 `load_digits` images (64 features).

- **Approach:**
  1. Projects the data to 2-D with six methods: **PCA, Kernel PCA, LLE, t-SNE, Isomap, and Laplacian Eigenmaps**.
  2. Uses **cumulative percent variance (CPV)** to choose components. PCA needs 40 components and KPCA needs 72 to retain 95%.
  3. Compares three pipelines: `Scaler → KPCA → SVC`, `Scaler → LDA → SVC`, and `Scaler → SVC`.
- **AI 221 concepts:**
  - **Linear vs. non-linear dimensionality reduction**, and **manifold learning** (neighborhood graphs, geodesic distances, graph Laplacians).
  - **t-SNE**: Gaussian affinities in high-D vs. Student-t affinities in low-D, and why that separates clusters well (derived in the notebook).
  - **Linear Discriminant Analysis** as supervised projection (≤ C−1 components).
  - **Dimensionality reduction as preprocessing** for a classifier.
- **Result:** KPCA (sigmoid, 40 components) + SVC scored **99.17%**, beating SVC on the raw 64 features (98.61%).

### 5. Country Development Classifier
📁 [`ai221/country-development-classifier`](ai221/country-development-classifier) · *with Joel Paolo Salvador*

Groups 167 countries by HELP International development indicators (child mortality, income, GDP, life expectancy, and others) to identify which countries most need aid.

- **Approach:** K-means (elbow + silhouette), then hierarchical clustering with single, complete, average, and Ward linkage, then PCA to 2-D followed by K-means again. Isolation Forest flags outlier countries.
- **AI 221 concepts:**
  - **Unsupervised learning:** K-means and **agglomerative hierarchical clustering** with dendrograms and linkage criteria.
  - **Choosing k:** the elbow method vs. **silhouette score**, and why silhouette is inflated by degenerate, imbalanced clusters.
  - **PCA** before clustering.
  - **Anomaly detection with Isolation Forest.**
- **Result:** Ward linkage with k = 5 gives the most balanced, interpretable groups. Isolation Forest flags outliers such as Singapore/Luxembourg (exports, GDP) and Nigeria (inflation).

### 6. Breast Tumor Classifier
📁 [`ai221/breast-tumor-classifier`](ai221/breast-tumor-classifier)

Classifies tumors as benign or malignant from 30 cell-nucleus features in the **Wisconsin Diagnostic Breast Cancer** dataset (569 samples).

- **Approach:** `StandardScaler → LogisticRegression` pipeline. Coefficient magnitudes are ranked to find the most informative features, and the model is evaluated on both train and test sets.
- **AI 221 concepts:**
  - **Logistic regression** as a probabilistic linear classifier.
  - **Multicollinearity** (radius/perimeter/area) and how scaling and regularization handle it.
  - **Evaluation beyond accuracy:** precision, recall, F1, **false alarm rate**, and **ROC curve / AUC**.
  - **Clinical error costs:** false negatives (missed malignancies) matter most.
- **Result:** Test **AUC 0.9975**, F1 0.960, precision 98.4%, recall 93.8%. "Worst-case" features (the `*3` suffix) are the strongest predictors.

### 7. Laguna Lake Chlorophyll-A Prediction
📁 [`ai221/laguna-lake-chlorophyll-A-prediction`](ai221/laguna-lake-chlorophyll-A-prediction)

Predicts **Chlorophyll-a** levels in Laguna de Bay from nutrient measurements (NO₂, NO₃, NH₄, PO₄). Laguna de Bay supplies treated drinking water to Metro Manila, and algal blooms can cause shortages.

- **Approach:** Compares **Gaussian Process Regression** (RBF, Matérn, RationalQuadratic, DotProduct, and WhiteKernel kernels) with **Kernel Ridge Regression**, both tuned with `GridSearchCV`.
- **AI 221 concepts:**
  - **Gaussian Processes:** kernel choice as a prior over functions, noise modeling with `WhiteKernel`, and predictive uncertainty.
  - **Kernel Ridge Regression** and how it relates to GPR (same kernel machinery, no probabilistic output).
  - **Overfitting diagnosis** from the train/test R² gap.
- **Result:** GPR with a RationalQuadratic kernel reached **test R² 0.86 / RMSE 0.54**, vs. KRR's R² 0.66 / RMSE 0.85.

---

## AI 201: Artificial Intelligence

### 8. Food Classifier for Fitness Goals (AI 201)
📁 [`ai201/foodclassifier`](ai201/foodclassifier) · *with Joel Paolo C. Salvador* · 📄 [Project report (PDF)](ai201/foodclassifier/Quiwa_Salvador_FoodClassifier_Report/Quiwa_Salvador_FoodClassifier.pdf)

Classifies foods as best suited for **muscle gain**, **weight loss**, or **endurance** based on their nutrient profiles (Kaggle Food Nutrition Dataset). Labels were assigned with thresholds taken from the nutrition literature.

- **Approach:** Four classifiers are tuned with `GridSearchCV` and compared using **5-fold cross-validation** on **macro-F1**. A one-way **ANOVA F-test** (`f_classif`) and logistic regression pick significant nutrients, and each model is then retrained on this extended feature set.
- **AI 201 concepts:**
  - **Probabilistic classification:** Gaussian Naive Bayes and its feature-independence assumption.
  - **Maximum-margin classifiers** (SVM with RBF kernel).
  - **Neural networks:** the multilayer perceptron.
  - **Ensemble learning / boosting:** AdaBoost (SAMME) reweights misclassified examples, which suits threshold-based labels.
  - **Statistical feature selection** (ANOVA) and **choosing F1 over accuracy** for imbalanced classes.
  - Knowledge-driven labeling: encoding domain rules as class thresholds.
- **Result:**

  | Model | Accuracy | Precision | Recall | F1 |
  |---|---|---|---|---|
  | Naive Bayes | 0.910 | 0.894 | 0.896 | 0.892 |
  | SVM | 0.950 | 0.945 | 0.937 | 0.937 |
  | MLP | 0.968 | 0.962 | 0.962 | 0.962 |
  | **AdaBoost** | **0.985** | **0.987** | **0.980** | **0.983** |

- **Files:**

  | Path | Contents |
  |---|---|
  | [`AI201 Mini Proj (1).ipynb`](ai201/foodclassifier/AI201%20Mini%20Proj%20%281%29.ipynb) | EDA, labeling, ANOVA, logistic regression, NB/SVM baseline |
  | [`AI201 Mini Proj - Extended Features.ipynb`](ai201/foodclassifier/AI201%20Mini%20Proj%20-%20Extended%20Features.ipynb) | NB/SVM on ANOVA-selected features |
  | [`foodclassifer_naive_svm.ipynb`](ai201/foodclassifier/foodclassifer_naive_svm.ipynb) | Naive Bayes and SVM with grid search |
  | [`foodclassifier_mlp.ipynb`](ai201/foodclassifier/foodclassifier_mlp.ipynb) / [`_extended`](ai201/foodclassifier/foodclassifier_mlp_extended.ipynb) | MLP (base / extended features) |
  | [`foodclassifier_adaboost.ipynb`](ai201/foodclassifier/foodclassifier_adaboost.ipynb) / [`_extended`](ai201/foodclassifier/foodclassifier_adaboost_extended.ipynb) | AdaBoost (base / extended features) |
  | [`PROJECT_DATA/`](ai201/foodclassifier/PROJECT_DATA) | Raw data by food group, per-group EDA plots, metadata |
  | [`Test Output Files/`](ai201/foodclassifier/Test%20Output%20Files) | Predictions on unlabeled foods |

---

## Concept index

Use this to find where a particular technique is used.

| Concept | Where to look |
|---|---|
| Linear / Ridge regression, regularization | [Energy Efficiency](ai221/energy-efficiency-in-buildings) |
| Logistic regression | [Breast Tumor](ai221/breast-tumor-classifier), [Food Classifier](ai201/foodclassifier) |
| Support Vector Machines / kernel trick | [Penguins](ai221/palmer-penguin-species-classifier), [Wine](ai221/wine-quality), [Digits](ai221/handwritten-digits), [Food Classifier](ai201/foodclassifier) |
| Kernel Ridge & Gaussian Process Regression | [Laguna Lake](ai221/laguna-lake-chlorophyll-A-prediction) |
| Naive Bayes | [Food Classifier](ai201/foodclassifier) |
| Neural networks (MLP) | [Wine](ai221/wine-quality), [Food Classifier](ai201/foodclassifier) |
| Ensembles: Random Forest, Gradient Boosting, AdaBoost | [Wine](ai221/wine-quality), [Food Classifier](ai201/foodclassifier) |
| PCA / Kernel PCA / LDA | [Digits](ai221/handwritten-digits), [Country Development](ai221/country-development-classifier) |
| Manifold learning (t-SNE, Isomap, LLE, Laplacian Eigenmaps) | [Digits](ai221/handwritten-digits) |
| Clustering (K-means, hierarchical) | [Country Development](ai221/country-development-classifier) |
| Anomaly detection (Isolation Forest) | [Country Development](ai221/country-development-classifier) |
| Hyperparameter tuning & cross-validation | [Wine](ai221/wine-quality), [Laguna Lake](ai221/laguna-lake-chlorophyll-A-prediction), [Food Classifier](ai201/foodclassifier) |
| Feature importance / selection | [Energy Efficiency](ai221/energy-efficiency-in-buildings), [Breast Tumor](ai221/breast-tumor-classifier), [Food Classifier](ai201/foodclassifier) |
| Class imbalance, F1, ROC/AUC | [Wine](ai221/wine-quality), [Breast Tumor](ai221/breast-tumor-classifier), [Food Classifier](ai201/foodclassifier) |

---

## Running the notebooks

```bash
git clone https://github.com/quielq/quielq-ml-projects.git
cd quielq-ml-projects
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
jupyter lab
```

Each project folder contains its own dataset, so notebooks run from inside their folder without extra downloads. The digits project loads its data directly from scikit-learn.

## Author

**Quiel Andrew I. Quiwa**. Some exercises were done with **Joel Paolo C. Salvador**, as noted above.
