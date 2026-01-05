# Obesity Risk Prediction ML Model

A machine learning project for multi-class classification of obesity levels based on lifestyle, dietary habits, and physical attributes. This project is designed for the Kaggle competition on obesity risk prediction.

## Project Overview

This repository contains a complete pipeline for predicting obesity categories using ensemble machine learning methods. The model classifies individuals into seven weight categories based on features like age, weight, height, eating habits, physical activity, and family history.

### Target Classes

- Insufficient_Weight
- Normal_Weight
- Overweight_Level_I
- Overweight_Level_II
- Obesity_Type_I
- Obesity_Type_II
- Obesity_Type_III

## Repository Structure

| File | Description |
|------|-------------|
| `train.csv` | Training dataset with 1996 samples and 19 features including the target variable |
| `test.csv` | Test dataset with 856 samples for prediction |
| `sample_submission.csv` | Template file showing the expected submission format |
| `Kaggle_Obesity_Full_Setup.py` | Standalone Python script with baseline, stacking, and hyperparameter tuning modes |
| `OctWave_Obesity_Trainer.ipynb` | Main Jupyter notebook with EDA, feature engineering, and ensemble training |
| `Untitled0.ipynb` | Extended notebook with additional models (LightGBM, CatBoost) and advanced feature engineering |
| `octwave_fixed_sub.csv` | Generated submission file with predictions |

## Dataset Features

### Numerical Features
- Age_Years
- Weight_Kg
- Height_cm
- Vegetable_Intake
- Meal_Frequency
- Water_Intake
- Screen_Time_Hours
- Family_Risk
- Activity_Level_Score

### Categorical Features
- Gender
- High_Calorie_Food (yes/no)
- Family_History (yes/no)
- Snack_Frequency
- Smoking_Habit
- Alcohol_Consumption
- Commute_Mode
- Physical_Activity_Level
- Leisure Time Activity

## Models and Methods

### Kaggle_Obesity_Full_Setup.py

Command-line script supporting three modes:

1. **Baseline Mode**: XGBoost classifier with 5-fold stratified cross-validation
2. **Stack Mode**: Stacking ensemble combining Random Forest, Gradient Boosting, and XGBoost
3. **Tune Mode**: Optuna-based hyperparameter optimization for XGBoost

Usage:
```
python Kaggle_Obesity_Full_Setup.py --mode baseline
python Kaggle_Obesity_Full_Setup.py --mode stack
python Kaggle_Obesity_Full_Setup.py --mode tune --n_trials 50
```

### OctWave_Obesity_Trainer.ipynb

Interactive notebook implementing:

- Exploratory data analysis with correlation heatmaps
- Data preprocessing and label encoding
- Feature engineering (BMI calculation, Age-BMI interaction)
- SMOTE for handling class imbalance
- Ensemble of Random Forest, Gradient Boosting, and Extra Trees with soft voting
- Model comparison against standalone XGBoost

### Untitled0.ipynb

Extended pipeline featuring:

- Additional models: LightGBM, CatBoost
- Advanced feature engineering: BMI squared, Sedentary score, Activity-Water interaction
- Comprehensive data cleaning with Yes/No normalization
- GPU acceleration support for XGBoost

## Feature Engineering

Key engineered features that improve model performance:

| Feature | Formula |
|---------|---------|
| BMI | Weight_Kg / (Height_cm / 100)^2 |
| Age_BMI | Age_Years * BMI |
| BMI_Squared | BMI^2 |
| Sedentary_Score | Screen_Time_Hours - Activity_Level_Score |
| Activity_Water | Activity_Level_Score * Water_Intake |

## Dependencies

- numpy
- pandas
- scikit-learn
- xgboost
- lightgbm
- catboost
- imbalanced-learn
- optuna
- matplotlib
- seaborn

Install all dependencies:
```
pip install numpy pandas scikit-learn xgboost lightgbm catboost imbalanced-learn optuna matplotlib seaborn
```

## How to Run

1. Place `train.csv` and `test.csv` in the project directory
2. Run the Python script or open a notebook
3. For notebooks, execute cells sequentially
4. The submission file will be generated as `octwave_fixed_sub.csv`

## Model Performance

The ensemble approach with SMOTE balancing achieves competitive cross-validation accuracy. The weighted voting classifier combines multiple tree-based models to improve generalization.

## Notes

- The dataset contains some inconsistent values in categorical columns (e.g., "yess" instead of "yes") which are handled during preprocessing
- GPU acceleration is available for XGBoost when running on systems with NVIDIA GPUs
- SMOTE oversampling is applied to handle class imbalance in the training data
