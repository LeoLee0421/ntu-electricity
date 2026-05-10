# Import packages ------------------------------------------------------------------------------
import requests
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import time
from tqdm import tqdm
import os
import glob
import traceback
import io
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import KFold
import statsmodels.api as sm


# Set paths ------------------------------------------------------------------------------------
WORK_DIR = os.getcwd()
PROCESSED_FOLDER = os.path.join(WORK_DIR, "data", "processed", "alltime")
RESULT_FOLDER = os.path.join(WORK_DIR, "data", "results", "alltime")

# 建築對應（中文 → 英文）
BUILDING_MAP = {
    "普通": "putong",
    "綜合": "zonghe",
    "博雅": "boya",
    "新生": "xinsheng",
    "共同": "gongtong"
}


# Linear regression -------------------------------------------------------------------------------

# Initialize list to store statistical results
results_list = []

# Define semester periods based on user comments
train_period_1 = ('2024-09-02', '2024-12-20')
train_period_2 = ('2025-02-24', '2025-06-06')
test_period = ('2025-09-01', '2025-12-19')

for ch_name, en_name in BUILDING_MAP.items():
    
    # Load data
    file_path = os.path.join(PROCESSED_FOLDER, f"{en_name}_final_combined_alltime.csv")
    if not os.path.exists(file_path):
        continue
        
    df = pd.read_csv(file_path)
    df['DateTime'] = pd.to_datetime(df['DateTime'])

    # Split dataset
    # Training set: First two semesters
    train_mask = (
        ((df['DateTime'] >= train_period_1[0]) & (df['DateTime'] <= train_period_1[1])) |
        ((df['DateTime'] >= train_period_2[0]) & (df['DateTime'] <= train_period_2[1]))
    )
    # Test set: Last semester
    test_mask = (df['DateTime'] >= test_period[0]) & (df['DateTime'] <= test_period[1])
    
    df_train = df[train_mask].copy()
    df_test = df[test_mask].copy()

    if df_train.empty or df_test.empty:
        print(f"Warning: Insufficient data for {en_name}, skipping.")
        continue

    # Training
    features = ["Temp", "BigC", "MediumC", "SmallC"]
    X_train = sm.add_constant(df_train[features]) 
    y_train = df_train["Electricity"]
    
    # Fit OLS model using statsmodels
    res = sm.OLS(y_train, X_train).fit()
    print(res.summary())

    # Save data: Coefficients, Standard Errors, and R-squared
    for feature in ['const'] + features:
        results_list.append({
            'Building': en_name,
            'Variable': feature,
            'Coef': res.params[feature],
            'StdErr': res.bse[feature],
            'R2': res.rsquared
        })

    # Predict model
    X_test = sm.add_constant(df_test[features])
    y_test = df_test["Electricity"]
    y_pred = res.predict(X_test)

    # Instead of plotting, save the comparison as a table
    prediction_df = pd.DataFrame({
        'DateTime': df_test['DateTime'],
        'Actual_Electricity': y_test,
        'Predicted_Electricity': y_pred,
        'Residual': y_test - y_pred
    })
    
    # Save test prediction comparison CSV
    pred_filename = f"{en_name}_test_prediction_comparison.csv"
    prediction_df.to_csv(os.path.join(RESULT_FOLDER, pred_filename), index=False)
    print(f"Saved prediction table for {ch_name}: {pred_filename}")

# Final output: Summary CSV
results_df = pd.DataFrame(results_list)
results_df.to_csv(os.path.join(RESULT_FOLDER, "regression_results_alltime.csv"), index=False)

print("\nProcess completed successfully.")
print("1. Statistical summary saved to regression_results_alltime.csv")
print("2. Individual building test predictions saved to separate CSV files.")