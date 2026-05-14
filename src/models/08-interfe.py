import pandas as pd
import numpy as np
import os
import statsmodels.api as sm

# ── Paths ──────────────────────────────────────────────────────────────────────
WORK_DIR         = os.getcwd()
PROCESSED_FOLDER = os.path.join(WORK_DIR, "data", "processed", "alltime")
RESULT_FOLDER    = os.path.join(WORK_DIR, "data", "results")
os.makedirs(RESULT_FOLDER, exist_ok=True)

# ── Config ─────────────────────────────────────────────────────────────────────
BUILDING_MAP = {"普通": "putong", "新生": "xinsheng", "共同": "gongtong"}
NTU_HOLIDAYS = [
    "2024-09-17", "2024-10-02", "2024-10-03", "2024-10-04", "2024-10-10",
    "2024-10-31", "2024-11-22",
    "2025-02-28", "2025-04-03", "2025-04-04", "2025-05-30",
    "2025-09-29", "2025-10-06", "2025-10-10", "2025-10-24", "2025-11-21"
]
HOLIDAY_TS     = pd.to_datetime(NTU_HOLIDAYS).normalize()
TARGET_PERIODS = [str(i) for i in range(1, 11)]

# 各棟 BigC 上限（普通=4, 新生=3, 共同=3）
BIGC_MAX = {'putong': 4, 'xinsheng': 3, 'gongtong': 3}

# ── Model Builder ──────────────────────────────────────────────────────────────
def build_features(df):
    df = df.copy()
    df['Temp_c']          = df['Temp'] - 25
    df['Temp_c_x_BigC']   = df['Temp_c'] * df['BigC']
    df['Temp_c_x_SmallC'] = df['Temp_c'] * df['SmallC']
    df['month']           = df['DateTime'].dt.month
    df['hour']            = df['DateTime'].dt.hour

    month_dummies = pd.get_dummies(df['month'], prefix='month', drop_first=True).astype(int)
    hour_dummies  = pd.get_dummies(df['hour'],  prefix='hour',  drop_first=True).astype(int)

    base = ['Temp_c', 'BigC', 'SmallC', 'Temp_c_x_BigC', 'Temp_c_x_SmallC']
    X = pd.concat([df[base], month_dummies, hour_dummies], axis=1)
    return sm.add_constant(X)


# ── Main Loop ──────────────────────────────────────────────────────────────────
for ch_name, en_name in BUILDING_MAP.items():
    file_path = os.path.join(PROCESSED_FOLDER, f"{en_name}_final_combined_alltime.csv")
    if not os.path.exists(file_path):
        print(f"[SKIP] 找不到檔案: {file_path}")
        continue

    df = pd.read_csv(file_path)
    df['DateTime'] = pd.to_datetime(df['DateTime'])
    df['Period']   = df['Period'].astype(str).str.replace('.0', '', regex=False)
    is_not_holiday = ~df['DateTime'].dt.normalize().isin(HOLIDAY_TS)

    # ── Masks ──
    train_mask = (
        ((df['DateTime'] >= '2024-09-02') & (df['DateTime'] <= '2024-12-20'))
        #|((df['DateTime'] >= '2025-02-24') & (df['DateTime'] <= '2025-06-06'))
    ) & is_not_holiday & df['Period'].isin(TARGET_PERIODS)

    test_mask = (
        (df['DateTime'] >= '2025-09-01') & (df['DateTime'] <= '2025-12-19')
    ) & is_not_holiday & df['Period'].isin(TARGET_PERIODS)

    if en_name == "putong":
        bad = (df['DateTime'] >= '2025-09-01') & (df['DateTime'] <= '2025-10-31')
        test_mask = test_mask & ~bad
        print(f"[INFO] {en_name}: 排除 2025-09~10 測試資料")

    df_train = df[train_mask].copy().reset_index(drop=True)
    df_test  = df[test_mask].copy().reset_index(drop=True)
    if df_train.empty or df_test.empty:
        print(f"[SKIP] {en_name}: 訓練或測試資料為空")
        continue

    # ── Fit ──
    X_train     = build_features(df_train)
    y_train     = df_train['Electricity']
    model_fit   = sm.OLS(y_train, X_train).fit()

    X_test = build_features(df_test).reindex(columns=X_train.columns, fill_value=0)
    y_pred = model_fit.predict(X_test)

    # ── Metrics ──
    rmse_oos   = np.sqrt(((df_test['Electricity'] - y_pred) ** 2).mean())
    r2_oos     = float(np.corrcoef(df_test['Electricity'], y_pred)[0, 1] ** 2)
    r2_in      = model_fit.rsquared
    rmse_in    = np.sqrt(model_fit.mse_resid)

    # 1. 預測結果
    pred_df = pd.DataFrame({
        'DateTime' : df_test['DateTime'],
        'Actual'   : df_test['Electricity'].values,
        'Predicted': y_pred.values,
        'Residual' : (df_test['Electricity'].values - y_pred.values)
    })
    pred_df.to_csv(os.path.join(RESULT_FOLDER, f"{en_name}_predictions.csv"), index=False)

    # 2. 模型效能指標
    metrics_df = pd.DataFrame([{
        'Building'  : en_name,
        'R2_in'     : round(r2_in,   4),
        'RMSE_in'   : round(rmse_in,  4),
        'R2_oos'    : round(r2_oos,   4),
        'RMSE_oos'  : round(rmse_oos, 4),
        'N_train'   : int(model_fit.nobs),
        'N_test'    : len(df_test),
    }])
    metrics_df.to_csv(os.path.join(RESULT_FOLDER, f"{en_name}_model_metrics.csv"), index=False)

    # 3. 係數表（含 p-value, std err）
    coef_df = pd.DataFrame({
        'Variable': model_fit.params.index,
        'Coef'    : model_fit.params.values,
        'StdErr'  : model_fit.bse.values,
        'P_value' : model_fit.pvalues.values,
    })
    coef_df.to_csv(os.path.join(RESULT_FOLDER, f"{en_name}_coefficients.csv"), index=False)

    # 4. 情境模擬：少開 N 間大教室省多少電
    # 邏輯：BigC 減少 N，重新預測，差值就是省下的電量
    # 上限由 BIGC_MAX 決定，各棟不同
    scenario_rows = []
    for n in range(1, BIGC_MAX[en_name] + 1):
        df_scenario          = df_test.copy()
        df_scenario['BigC']  = np.maximum(df_scenario['BigC'] - n, 0)  # 不能為負
        X_scenario           = build_features(df_scenario).reindex(columns=X_train.columns, fill_value=0)
        y_scenario           = model_fit.predict(X_scenario)

        # 省電量 (kWh)
        savings_per_slot     = y_pred.values - y_scenario.values
        total_savings_kwh    = savings_per_slot.sum()
        avg_savings_per_day  = savings_per_slot.mean() * 10             # 10節/天換算

        scenario_rows.append({
            'Reduce_BigC_N'        : n,
            'Total_Savings_kWh'    : round(total_savings_kwh,   2),
            'Avg_Savings_per_Day'  : round(avg_savings_per_day, 2),
            'Avg_Savings_per_Slot' : round(savings_per_slot.mean(), 4),
            'Test_Days'            : len(df_test['DateTime'].dt.normalize().unique()),
        })

    scenario_df = pd.DataFrame(scenario_rows)
    scenario_df.to_csv(os.path.join(RESULT_FOLDER, f"{en_name}_scenario_savings.csv"), index=False)

    print(f"[OK] {en_name}: In-R²={r2_in:.3f} | OOS-R²={r2_oos:.3f} | RMSE={rmse_oos:.2f}")

print("\n[完成] 所有模型結果已輸出至:", RESULT_FOLDER)