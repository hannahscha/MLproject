# **Student Grade Prediction Project**

This project aims to predict student grades based on their activities, demographic features, and other influential factors using machine learning models. The project employs multiple regression models, evaluates them using various metrics, and optimizes performance through hyperparameter tuning.

## **Table of Contents**
1. [Introduction](#introduction)
2. [Features](#features)
3. [Project Workflow](#project-workflow)
4. [Models and Evaluation](#models-and-evaluation)
5. [Ethics](#ethics)
6. [Important Figures](#important-figures)

---

## **Introduction**
This project focuses on leveraging machine learning techniques to predict student grades based on their activities and personal attributes. It explores the relationships between input features and student performance to guide interventions for better academic outcomes.

### **Objective**
- To develop and compare regression models to predict student grades.
- To analyze model performance using metrics like **Accuracy**, **R²**, **RMSE**, and **MAE**.
- Creat a user freindly app utilizing the model.

---

## **Features**
The dataset includes:
- **Categorical Features**: Encoded for use in the models (e.g., `school`, `sex`, `address`).
- **Numerical Features**: Such as `studytime`, `failures`, and `absences`.
- **Target Variable**: The final grade of the student.

Key transformations include:
- Stratified sampling based on `Medu` and `failures`.
- One-hot encoding for categorical variables.

---

## **Project Workflow**
1. **Data Preprocessing**
   - Handling missing values.
   - Feature scaling and encoding.
   - Train-test split using `StratifiedShuffleSplit`.

2. **Model Development**
   - Linear Regression.
   - Random Forest, Decision Tree, Gradient Boosting, and others.
   - Hyperparameter tuning for optimal performance.

3. **Evaluation**
   - Metrics used: **Accuracy (R²)**, **RMSE**, and **MAE** across Train, Validation, and Test sets.

4. **Results Compilation**
   - Results saved to `model_results.csv`.

---

## **Models and Evaluation**
The following models were trained and evaluated:
- **Linear Regression**
- **Random Forest**
- **Decision Tree**
- **Gradient Boosting**
- **K-Nearest Neighbors**
- **Support Vector Machine**
- **Artificial Neural Networks**
- **Feature-Based Regression (Ridge)**
- **XGBoost**
- **LightGBM**
- **CatBoost**

The best model was determined to be the Random Forest model cross-validated with Bayesian Search to get the following best performance hyperparameters:
```python

    ('max_depth', 10),
    ('min_samples_leaf', 2),
    ('min_samples_split', 6),
    ('n_estimators', 50)

```

The model was saved as a .pkl file (best_bayesian_random_forest_model.pkl) to be used in the app.

---

## **Ethics**
Ethical considerations including Deon Ethics Checklist and Ethics Data Card Checklist were used to evaluate the social impact of the model and app. These are available through the following streamlit app links:

Deon Ethics:
https://mldeonethics-jtppmpppkpa5bjjjkws6pm.streamlit.app/

Ethics Data Card:
https://mlethicsdatacard-njdaykz4wx5vxcbcnd8twf.streamlit.app/

---

## **Important Figures**
Through trining the model we only considered the features with over 0.05 correlation to the model. To understand any internal biases of the dataset, the distribution, bee swarm plot, and partial dependence plots are shown below.
![Feature Distruibution](feature_distribution.png)
![Bee Swarm Plot](bee_swarm_plot.png)
![Partial Dependence Plot](partial_dependence_plot.png)
