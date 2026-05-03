# Import packages ------------------------------------------------------------------------------
import requests
import pandas as pd
import numpy as np
import os
import statsmodels.api as sm


# Set paths ------------------------------------------------------------------------------------
WORK_DIR = os.getcwd()
PROCESSED_FOLDER = os.path.join(WORK_DIR, "data", "processed")
RESULT_FOLDER = os.path.join(WORK_DIR, "data", "results", "113-1-subset")

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
train_period_2 = ('2024-09-02', '2024-12-20')
#train_period_2 = ('2025-02-24', '2025-06-06')
#train_period_2 = ('2025-09-01', '2025-12-19')
test_period = ('2025-09-01', '2025-12-19')

# Define daytime periods (1-10)
daytime_periods = [str(i) for i in range(1, 11)]

for ch_name, en_name in BUILDING_MAP.items():
    
    # Load data
    file_path = os.path.join(PROCESSED_FOLDER, f"{en_name}_final_combined.csv")
    if not os.path.exists(file_path):
        continue
        
    df = pd.read_csv(file_path)
    df['DateTime'] = pd.to_datetime(df['DateTime'])

    # --- 新增限制：過濾 Period 僅保留 1 到 10 ---
    # 確保 Period 欄位為字串型態後進行比對
    #df = df[df['Period'].astype(str).isin(daytime_periods)].copy()

    # Split dataset
    # Training set: First two semesters
    train_mask = (
        #((df['DateTime'] >= train_period_1[0]) & (df['DateTime'] <= train_period_1[1])) |
        ((df['DateTime'] >= train_period_2[0]) & (df['DateTime'] <= train_period_2[1]))
    )
    # Test set: Last semester
    test_mask = (df['DateTime'] >= test_period[0]) & (df['DateTime'] <= test_period[1])
    
    df_train = df[train_mask].copy()
    df_test = df[test_mask].copy()

    if df_train.empty or df_test.empty:
        print(f"Warning: Insufficient data for {en_name} after daytime filtering, skipping.")
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
        'Period': df_test['Period'], # 新增 Period 資訊方便核對
        'Actual_Electricity': y_test,
        'Predicted_Electricity': y_pred,
        'Residual': y_test - y_pred
    })
    
    # Save test prediction comparison CSV
    #pred_filename = f"114-1_subset_{en_name}_test_prediction_comparison.csv"
    #prediction_df.to_csv(os.path.join(RESULT_FOLDER, pred_filename), index=False)
    #print(f"Saved prediction table for {ch_name}: {pred_filename}")

# Final output: Summary CSV
results_df = pd.DataFrame(results_list)
results_df.to_csv(os.path.join(RESULT_FOLDER, "113-1_subset.csv"), index=False)

print("\nProcess completed successfully with Daytime (Period 1-10) filter.")