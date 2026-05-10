import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import statsmodels.api as sm
import matplotlib.font_manager as fm

# Font & Plot Settings --------------------------------------------------------------------------
_CJK_CANDIDATES = ['PingFang TC', 'Heiti TC', 'Noto Sans CJK TC', 'Microsoft JhengHei']
_available = {f.name for f in fm.fontManager.ttflist}
_cjk_font = next((f for f in _CJK_CANDIDATES if f in _available), None)
if _cjk_font: plt.rcParams['font.family'] = _cjk_font
plt.rcParams['axes.unicode_minus'] = False

# Paths -----------------------------------------------------------------------------------------
WORK_DIR = os.getcwd()
PROCESSED_FOLDER = os.path.join(WORK_DIR, "data", "processed", "alltime")
RESULT_FOLDER = os.path.join(WORK_DIR, "data", "results")
PLOT_FOLDER = os.path.join(WORK_DIR, "visualization")
os.makedirs(RESULT_FOLDER, exist_ok=True)
os.makedirs(PLOT_FOLDER, exist_ok=True)

# Config ----------------------------------------------------------------------------------------
BUILDING_MAP = {"普通": "putong", "新生": "xinsheng", "共同": "gongtong"}
NTU_HOLIDAYS = ["2024-09-17", "2024-10-02", "2024-10-03", "2024-10-04", "2024-10-10", "2024-10-31", "2024-11-22", 
                "2025-02-28", "2025-04-03", "2025-04-04", "2025-05-30", 
                "2025-09-29", "2025-10-06", "2025-10-10", "2025-10-24", "2025-11-21"] 
HOLIDAY_TS = pd.to_datetime(NTU_HOLIDAYS).normalize()
TARGET_PERIODS = ["1", "2", "3", "4", "5", "6", "7", "8", "9", "10"]

# Model Construction ----------------------------------------------------------------------------
def build_advanced_model(df):
    """
    Combined Model: 
    1. Interaction: (Temp - 25) * Classroom
    2. Non-linear: Temp^2
    3. Fixed Effects: Month, Weekday, Hour
    """
    df = df.copy()
    
    # 1. Interaction & Centering
    df['Temp_c'] = df['Temp'] - 25
    df['Temp2'] = df['Temp'] ** 2
    df['Temp_c_x_BigC'] = df['Temp_c'] * df['BigC']
    df['Temp_c_x_SmallC'] = df['Temp_c'] * df['SmallC']
    
    # 2. Time Features
    df['month'] = df['DateTime'].dt.month
    df['weekday'] = df['DateTime'].dt.weekday
    df['hour'] = df['DateTime'].dt.hour
    
    # 3. Create Dummies
    month_dummies = pd.get_dummies(df['month'], prefix='month', drop_first=True).astype(int)
    weekday_dummies = pd.get_dummies(df['weekday'], prefix='weekday', drop_first=True).astype(int)
    hour_dummies = pd.get_dummies(df['hour'], prefix='hour', drop_first=True).astype(int)
    
    # Combine all
    base_features = ['Temp_c', 'BigC', 'SmallC', 'Temp_c_x_BigC', 'Temp_c_x_SmallC']
    X = pd.concat(
        [df[base_features], 
         month_dummies, 
         #weekday_dummies, 
         hour_dummies], 
        axis=1)
    X = sm.add_constant(X)
    
    return X

# Main Execution --------------------------------------------------------------------------------
all_coeffs = []

for ch_name, en_name in BUILDING_MAP.items():
    file_path = os.path.join(PROCESSED_FOLDER, f"{en_name}_final_combined_alltime.csv")
    if not os.path.exists(file_path): continue

    df = pd.read_csv(file_path)
    df['DateTime'] = pd.to_datetime(df['DateTime'])
    
    # Filter: Non-Holiday & Class Periods
    is_not_holiday = ~df['DateTime'].dt.normalize().isin(HOLIDAY_TS)
    
    train_mask = (
        ((df['DateTime'] >= '2024-09-02') & (df['DateTime'] <= '2024-12-20')) 
        #| ((df['DateTime'] >= '2025-02-24') & (df['DateTime'] <= '2025-06-06'))
    ) & is_not_holiday & df['Period'].isin(TARGET_PERIODS)

    test_mask = (
        (df['DateTime'] >= '2025-09-01') & (df['DateTime'] <= '2025-12-19')
    ) & is_not_holiday & df['Period'].isin(TARGET_PERIODS)

    df_train = df[train_mask].copy().reset_index(drop=True)
    df_test = df[test_mask].copy().reset_index(drop=True)

    if df_train.empty or df_test.empty: continue

    # Fit Model
    X_train = build_advanced_model(df_train)
    y_train = df_train["Electricity"]
    model_fit = sm.OLS(y_train, X_train).fit()

    # Predict
    X_test = build_advanced_model(df_test)
    X_test = X_test.reindex(columns=X_train.columns, fill_value=0)
    y_pred = model_fit.predict(X_test)

    # Metrics
    rmse = np.sqrt(((df_test['Electricity'] - y_pred) ** 2).mean())
    r2_oos = np.corrcoef(df_test['Electricity'], y_pred)[0, 1] ** 2

    # Save Results
    res_df = pd.DataFrame({
        'DateTime': df_test['DateTime'],
        'Actual': df_test['Electricity'],
        'Predicted': y_pred
    })
    res_df.to_csv(os.path.join(RESULT_FOLDER, f"{en_name}_predictions.csv"), index=False)

    # --- Visualization ---
    _title = ch_name if _cjk_font else en_name
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(15, 10), gridspec_kw={'height_ratios': [2, 1]})
    
    # 1. Time Series (Daily Mean to skip holidays/weekends smoothly)
    plot_df = res_df.set_index('DateTime').resample('D').mean().dropna().reset_index()
    
    ax1.plot(plot_df['DateTime'], plot_df['Actual'], color='#2c2c2c', label='Actual', alpha=0.7)
    ax1.plot(plot_df['DateTime'], plot_df['Predicted'], color='#6ab187', linestyle='--', label='Predicted')
    ax1.set_title(f"{_title} 電力預測 (交乘項 + 固定效應)\nOut-of-Sample R²: {r2_oos:.3f} | RMSE: {rmse:.2f}")
    ax1.legend()
    ax1.grid(True, linestyle=':', alpha=0.6)
    ax1.xaxis.set_major_formatter(mdates.DateFormatter('%m-%d'))
    
    # 2. Scatter
    ax2.scatter(df_test['Electricity'], y_pred, alpha=0.2, s=10, color='#6ab187')
    ax2.plot([y_train.min(), y_train.max()], [y_train.min(), y_train.max()], 'r--')
    ax2.set_xlabel("Actual Electricity (kWh)")
    ax2.set_ylabel("Predicted Electricity (kWh)")
    
    plt.tight_layout()
    plt.savefig(os.path.join(PLOT_FOLDER, f"{en_name}_final_eval.png"), dpi=200)
    plt.close()

    print(f"Finished {en_name}: R² = {r2_oos:.3f}")

results_list = [] # 用於儲存每一行回歸結果

for ch_name, en_name in BUILDING_MAP.items():
    file_path = os.path.join(PROCESSED_FOLDER, f"{en_name}_final_combined_alltime.csv")
    if not os.path.exists(file_path): continue

    df = pd.read_csv(file_path)
    df['DateTime'] = pd.to_datetime(df['DateTime'])
    
    # Filter 邏輯維持不變
    is_not_holiday = ~df['DateTime'].dt.normalize().isin(HOLIDAY_TS)
    train_mask = (((df['DateTime'] >= '2024-09-02') & (df['DateTime'] <= '2024-12-20')) |
                  ((df['DateTime'] >= '2025-02-24') & (df['DateTime'] <= '2025-06-06'))) & is_not_holiday & df['Period'].isin(TARGET_PERIODS)
    test_mask = ((df['DateTime'] >= '2025-09-01') & (df['DateTime'] <= '2025-12-19')) & is_not_holiday & df['Period'].isin(TARGET_PERIODS)

    df_train = df[train_mask].copy().reset_index(drop=True)
    df_test = df[test_mask].copy().reset_index(drop=True)
    if df_train.empty or df_test.empty: continue

    # Fit Model
    X_train = build_advanced_model(df_train)
    y_train = df_train["Electricity"]
    model_fit = sm.OLS(y_train, X_train).fit()

    # --- 新增：儲存回歸係數與統計量 ---
    for var_name, coef_val in model_fit.params.items():
        results_list.append({
            'Building': en_name,
            'Variable': var_name,
            'Coef': coef_val,
            'P_value': model_fit.pvalues[var_name], # 額外增加 p-value 供參考
            'R2': model_fit.rsquared,
            'N_train': int(model_fit.nobs)
        })

    # Predict & Visualization 邏輯維持不變 ...
    # (此處省略中間的 X_test 預測與 plt 繪圖程式碼，請保留您原本的內容)
    X_test = build_advanced_model(df_test)
    X_test = X_test.reindex(columns=X_train.columns, fill_value=0)
    y_pred = model_fit.predict(X_test)
    # ... (plt.savefig 等繪圖區塊) ...

    print(f"Finished {en_name}: R² = {model_fit.rsquared:.3f}")

# ── 輸出回歸資料區塊 ──────────────────────────────────────────────────────────

if results_list:
    results_df = pd.DataFrame(results_list)

    # 1. 儲存模型整體統計量 (R2, N)
    model_stats = results_df[['Building', 'R2', 'N_train']].drop_duplicates()
    stats_path = os.path.join(RESULT_FOLDER, "advanced_model_stats.csv")
    model_stats.to_csv(stats_path, index=False)

    # 2. 儲存詳細係數表 (Pivot 展開，讓不同建築並列)
    coef_wide = results_df.pivot(
        index='Variable',
        columns='Building',
        values='Coef'
    )
    
    # 將 Constant 排在第一列（如有）
    if 'const' in coef_wide.index:
        idx = ['const'] + [i for i in coef_wide.index if i != 'const']
        coef_wide = coef_wide.reindex(idx)

    summary_path = os.path.join(RESULT_FOLDER, "advanced_model_coefficients.csv")
    coef_wide.to_csv(summary_path)

    print(f"\n[成功] 係數摘要已儲存至: {summary_path}")
    print(f"[成功] 模型統計量已儲存至: {stats_path}")

print("All tasks completed.")