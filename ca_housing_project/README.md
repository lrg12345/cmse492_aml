# California Housing ML Project

This repository contains a complete machine learning workflow for predicting California housing prices. It includes data processing, exploratory analysis, feature engineering, model training, cross-validation, hyperparameter tuning, and saving trained models.

---

## Repository Structure

/analysis
ida.ipynb # Initial Data Analysis (data loading, stratified splitting)
eda.ipynb # Exploratory Data Analysis (visualization, correlations, feature engineering)

/data
/train
housing_train.csv # Raw stratified training set (13 features)
housing_train_processed.csv # Processed training set (24 features)
/test
housing_test.csv # Raw testing set (13 features)

/datasets
/housing
housing.csv
housing.tgz

/models
LinearRegression.ipynb # Linear Regression model notebook
DecisionTree.ipynb # Decision Tree model notebook
RandomForest.ipynb # Random Forest model notebook
SVR.ipynb # Support Vector Regression model notebook
*.pkl # Trained models saved from each notebook

preprocessing_pipeline.py # Python script to preprocess raw training data


---

## Analysis Notebooks

### `ida.ipynb` – Initial Data Analysis
- Loads raw California housing dataset.
- Performs data type analysis.
- Splits the dataset into **stratified training and testing sets**.
- Saves raw and processed datasets to `/data/train` and `/data/test`.

### `eda.ipynb` – Exploratory Data Analysis
- Conducts geographic data visualization.
- Calculates feature correlations.
- Performs feature engineering to create additional features (total of 24 features).
- Saves the processed dataset to `/data/train`.

---

## Preprocessing Script

### `preprocessing_pipeline.py`
- Reads the raw training set.
- Applies a scikit-learn pipeline for transformations and feature engineering.
- Saves the processed training set (24 features) to `/data/train`.

---

## Models

Each model notebook includes the following sections:
1. **Data Loading** – Load processed training dataset.
2. **Model Fitting** – Initialize and fit the model.
3. **Cross-Validation** – Evaluate using cross-validation.
4. **Hyperparameter Tuning** – Optimize model parameters with GridSearchCV or RandomizedSearchCV.
5. **Model Saving** – Save the trained model (`.pkl`) to `/models`.

**Included Models:**
- `LinearRegression.ipynb`
- `DecisionTree.ipynb`
- `RandomForest.ipynb`
- `SVR.ipynb`

---

## Requirements

- Python 3.x
- pandas
- numpy
- scikit-learn
- matplotlib
- joblib

You can install required packages via:

```bash
pip install pandas numpy scikit-learn matplotlib joblib