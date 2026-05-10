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
import matplotlib.dates as mdates
import seaborn as sns
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import KFold
import statsmodels.api as sm
import matplotlib.font_manager as fm


# Font settings: use a CJK-compatible font if available, else fall back gracefully
# macOS: Heiti TC / PingFang TC  |  Linux: Noto Sans CJK TC  |  Windows: Microsoft JhengHei
_CJK_CANDIDATES = [
    'PingFang TC', 'Heiti TC', 'STHeiti',          # macOS
    'Noto Sans CJK TC', 'Noto Sans TC',             # Linux
    'Microsoft JhengHei', 'Microsoft YaHei',         # Windows
]
_available = {f.name for f in fm.fontManager.ttflist}
_cjk_font  = next((f for f in _CJK_CANDIDATES if f in _available), None)
if _cjk_font:
    plt.rcParams['font.family'] = _cjk_font
else:
    print("Warning: No CJK font found. Chinese characters in plot titles will be replaced with English names.")
plt.rcParams['axes.unicode_minus'] = False   # prevent minus sign rendering issue


# Set paths ------------------------------------------------------------------------------------
WORK_DIR = os.getcwd()
PROCESSED_FOLDER = os.path.join(WORK_DIR, "data", "processed", "alltime")
RESULT_FOLDER = os.path.join(WORK_DIR, "data", "results", "adjust_models")
PLOT_FOLDER = os.path.join(WORK_DIR, "visualization", "predict_outcome")
os.makedirs(RESULT_FOLDER, exist_ok=True)
os.makedirs(PLOT_FOLDER, exist_ok=True)

# 建築對應與國定假日
BUILDING_MAP = {"普通": "putong", "新生": "xinsheng", "共同": "gongtong"}
NTU_HOLIDAYS = ["2024-09-17", "2024-10-02", "2024-10-03", "2024-10-04", "2024-10-10", "2024-10-31", "2024-11-22", 
                "2025-02-28", "2025-04-03", "2025-04-04", "2025-05-30", 
                "2025-09-29", "2025-10-06", "2025-10-10", "2025-10-24", "2025-11-21"] 

# Define semester periods ----------------------------------------------------------------------
train_period_1 = ('2024-09-02', '2024-12-20')
train_period_2 = ('2025-02-24', '2025-06-06')
test_period    = ('2025-09-01', '2025-12-19')

BASE_FEATURES = ["Temp", "BigC", "SmallC"]

# Model definitions ----------------------------------------------------------------------------
def build_model1(df):
    """Model 1: Baseline OLS (original features)"""
    features = BASE_FEATURES
    X = sm.add_constant(df[features])
    return X, ['const'] + features


def build_model2(df):
    """Model 2: Add quadratic temperature term (Temp²)"""
    df = df.copy()
    df['Temp2'] = df['Temp'] ** 2
    features = BASE_FEATURES + ['Temp2']
    X = sm.add_constant(df[features])
    return X, ['const'] + features


def build_model3(df):
    """Model 3: Add Temp x classroom interaction terms
    Temp is centered at 25 degrees C so that classroom main effects (BigC, SmallC)
    are interpreted as the marginal electricity use of one extra classroom at 25 degrees C.
    """
    df = df.copy()
    df['Temp_c']          = df['Temp'] - 25   # center Temp at 25 degrees C
    df['Temp_c_x_BigC']   = df['Temp_c'] * df['BigC']
    df['Temp_c_x_SmallC'] = df['Temp_c'] * df['SmallC']
    features = ['Temp_c', 'BigC', 'SmallC', 'Temp_c_x_BigC', 'Temp_c_x_SmallC']
    X = sm.add_constant(df[features])
    return X, ['const'] + features


def build_model4(df):
    """Model 4: Add time fixed effects (month, weekday, hour dummies)"""
    df = df.copy()
    df['month']   = df['DateTime'].dt.month
    df['weekday'] = df['DateTime'].dt.weekday   # 0=Monday ... 6=Sunday
    df['hour']    = df['DateTime'].dt.hour
    df['is_holiday'] = df['DateTime'].dt.normalize().isin(NTU_HOLIDAYS).astype(int)

    # Create dummies; drop_first=True avoids perfect multicollinearity
    month_dummies   = pd.get_dummies(df['month'],   prefix='month',   drop_first=True).astype(int)
    weekday_dummies = pd.get_dummies(df['weekday'], prefix='weekday', drop_first=True).astype(int)
    hour_dummies    = pd.get_dummies(df['hour'],    prefix='hour',    drop_first=True).astype(int)

    #X_base = df[BASE_FEATURES].reset_index(drop=True)
    X_all = pd.concat(
        [df[BASE_FEATURES],
         month_dummies,
         weekday_dummies,
         hour_dummies],
        axis=1
    )
    X = sm.add_constant(X_all)
    feature_names = list(X.columns)
    return X, feature_names


MODELS = [
    {'name': 'model1_baseline',      'label': 'Model 1: Baseline',      'build_fn': build_model1},
    {'name': 'model2_temp_squared',  'label': 'Model 2: Temp2',         'build_fn': build_model2},
    {'name': 'model3_interactions',  'label': 'Model 3: Interactions',  'build_fn': build_model3},
    {'name': 'model4_fixed_effects', 'label': 'Model 4: Fixed Effects', 'build_fn': build_model4},
]

# Plot colours (one per model + actual)
COLORS = {
    'actual':                '#2c2c2c',
    'model1_baseline':       '#e07b54',
    'model2_temp_squared':   '#5b8db8',
    'model3_interactions':   '#6ab187',
    'model4_fixed_effects':  '#b07cc6',
}

## Main loop ------------------------------------------------------------------------------------
results_list = []
pred_dict = {}

# 將字串列表轉換為 Timestamp 格式，方便後續比對
holiday_timestamps = pd.to_datetime(NTU_HOLIDAYS).normalize()

print("Running in PURE CLASS-TIME mode (Excluding Holidays from Train/Test).\n")

for ch_name, en_name in BUILDING_MAP.items():
    file_path = os.path.join(PROCESSED_FOLDER, f"{en_name}_final_combined_alltime.csv")
    if not os.path.exists(file_path): continue

    df = pd.read_csv(file_path)
    df['DateTime'] = pd.to_datetime(df['DateTime'])
    
    # 新增：建立一個「非假日」的 Mask
    # 使用 normalize() 確保只比對日期部分
    is_not_holiday = ~df['DateTime'].dt.normalize().isin(holiday_timestamps)

    # --- 修改後的 Semester Mask (加入非假日條件) ---
    train_mask = (
        ((df['DateTime'] >= '2024-09-02') & (df['DateTime'] <= '2024-12-20')) |
        ((df['DateTime'] >= '2025-02-24') & (df['DateTime'] <= '2025-06-06'))
    ) & is_not_holiday

    test_mask = (
        (df['DateTime'] >= '2025-09-01') & (df['DateTime'] <= '2025-12-19')
    ) & is_not_holiday

    # 強制只使用 Period 非空且非假日的資料
    df_train = df[train_mask & df['Period'].notna()].copy().reset_index(drop=True)
    df_test = df[test_mask & df['Period'].notna()].copy().reset_index(drop=True)

    if df_train.empty or df_test.empty: 
        print(f"Skipping {en_name}: No data after holiday filtering.")
        continue

    y_train = df_train["Electricity"]
    y_test = df_test["Electricity"]

    for model_cfg in MODELS:
        model_name, build_fn = model_cfg['name'], model_cfg['build_fn']
        try:
            X_train, feature_names = build_fn(df_train)
            res = sm.OLS(y_train, X_train).fit()

            # 儲存係數
            for feat in feature_names:
                if feat in res.params.index:
                    results_list.append({
                        'Building': en_name, 'Model': model_name, 'Variable': feat,
                        'Coef': res.params[feat], 'R2': res.rsquared, 'N_train': int(res.nobs)
                    })

            # 預測
            X_test, _ = build_fn(df_test)
            X_test = X_test.reindex(columns=X_train.columns, fill_value=0)
            y_pred = res.predict(X_test)

            prediction_df = pd.DataFrame({
                'DateTime': df_test['DateTime'],
                'Actual_Electricity': y_test.values,
                'Predicted_Electricity': y_pred.values,
                'Residual': (y_test - y_pred).values,
            })
            pred_dict[(en_name, model_name)] = prediction_df
            
            pred_filename = f"{en_name}_{model_name}_test_prediction.csv"
            prediction_df.to_csv(os.path.join(RESULT_FOLDER, pred_filename), index=False)
            print(f"Saved prediction CSV: {pred_filename}")     

        except Exception as e:
            print(f"Error in {model_name} for {en_name}: {e}")

# Plots ----------------------------------------------------------------------------------------
for ch_name, en_name in BUILDING_MAP.items():

    model_preds = [
        (m, pred_dict.get((en_name, m['name'])))
        for m in MODELS
        if (en_name, m['name']) in pred_dict
    ]
    if not model_preds:
        continue

    n_models = len(model_preds)
    _title_name = ch_name if _cjk_font else en_name

    # ── Figure 1: Time-series actual vs predicted (跳過假日，線條連續) ──────────────────
    fig, axes = plt.subplots(n_models, 1, figsize=(16, 4 * n_models), sharex=False)
    if n_models == 1:
        axes = [axes]

    fig.suptitle(
        f"{_title_name} ({en_name}) — 課堂時段電力預測與實際值 (已跳過非課堂日)",
        fontsize=13, fontweight='bold', y=1.01
    )

    for ax, (model_cfg, pdf) in zip(axes, model_preds):
        model_name  = model_cfg['name']
        model_label = model_cfg['label']
        color       = COLORS.get(model_name, '#888888')

        # --- 關鍵修改：resample 後立刻 dropna() ---
        # 這會移除所有沒有資料的日期，讓線條在畫圖時直接連起來
        pdf_plot = pdf.set_index('DateTime').resample('D').mean().dropna().reset_index()
        #pdf_plot = pdf.set_index('DateTime').dropna().reset_index()

        rmse = np.sqrt((pdf['Residual'] ** 2).mean())
        r2   = np.corrcoef(pdf['Actual_Electricity'], pdf['Predicted_Electricity'])[0, 1] ** 2

        # 繪製實際值與預測值
        ax.plot(pdf_plot['DateTime'], pdf_plot['Actual_Electricity'],
                color=COLORS['actual'], linewidth=1.5, label='Actual (Class Hours)', alpha=0.8)
        ax.plot(pdf_plot['DateTime'], pdf_plot['Predicted_Electricity'],
                color=color, linewidth=1.5, linestyle='--', label='Predicted (Class Hours)', alpha=0.9)

        # 設定標題與標籤
        ax.set_title(f"{model_label}   |   RMSE = {rmse:.2f}   Out-of-sample R² = {r2:.3f}",
                     fontsize=10, pad=5)
        ax.set_ylabel("Electricity (kWh)", fontsize=9)
        
        # 格式化 X 軸：雖然跳過了假日，但使用 DateFormatter 仍能顯示正確月份
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%m-%d'))
        ax.xaxis.set_major_locator(mdates.WeekdayLocator(byweekday=mdates.MO)) # 每週一顯示一個刻度
        ax.tick_params(axis='x', rotation=45, labelsize=8)
        
        ax.legend(fontsize=9, loc='upper right')
        ax.grid(True, linestyle=':', alpha=0.5)

    fig.tight_layout()
    ts_path = os.path.join(PLOT_FOLDER, f"{en_name}_timeseries_pure_classtime_continuous.png")
    fig.savefig(ts_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"Saved continuous time-series plot: {ts_path}")

    # ── Figure 2: Scatter (維持原樣) ──────────────────────────────────────────────────
    fig2, axes2 = plt.subplots(1, n_models, figsize=(5 * n_models, 5))
    if n_models == 1:
        axes2 = [axes2]

    fig2.suptitle(
        f"{_title_name} ({en_name}) — Scatter: Actual vs Predicted (Class Hours Only)",
        fontsize=13, fontweight='bold'
    )

    for ax2, (model_cfg, pdf) in zip(axes2, model_preds):
        model_name  = model_cfg['name']
        color       = COLORS.get(model_name, '#888888')

        actual    = pdf['Actual_Electricity']
        predicted = pdf['Predicted_Electricity']
        r2   = np.corrcoef(actual, predicted)[0, 1] ** 2

        ax2.scatter(actual, predicted, alpha=0.15, s=5, color=color)
        
        lim_min, lim_max = actual.min(), actual.max()
        ax2.plot([lim_min, lim_max], [lim_min, lim_max], color='#2c2c2c', linestyle='--')

        ax2.set_title(f"{model_cfg['label']}\nR² = {r2:.3f}", fontsize=9)
        ax2.set_xlabel("Actual (kWh)", fontsize=9)
        ax2.set_ylabel("Predicted (kWh)", fontsize=9)
        ax2.grid(True, linestyle=':', alpha=0.5)

    fig2.tight_layout()
    sc_path = os.path.join(PLOT_FOLDER, f"{en_name}_scatter_pure_classtime.png")
    fig2.savefig(sc_path, dpi=150, bbox_inches='tight')
    plt.close(fig2)


results_df = pd.DataFrame(results_list)

if not results_df.empty:
    # 1. 建立模型適配度摘要 (Model Fit Stats)
    # 提取每個模型唯一的 R2 和 N_train (不隨變數改變)
    model_stats = results_df[['Building', 'Model', 'R2', 'N_train']].drop_duplicates()
    
    # 轉為寬表格：列為 Building，欄為不同模型的統計量
    model_stats_wide = model_stats.pivot(
        index='Building', 
        columns='Model', 
        values=['R2', 'N_train']
    )
    # 重整欄位名稱，例如：model1_baseline_R2
    model_stats_wide.columns = [f"{mod}_{stat}" for stat, mod in model_stats_wide.columns]
    model_stats_wide = model_stats_wide.reset_index()
    
    stats_path = os.path.join(RESULT_FOLDER, "regression_model_stats.csv")
    model_stats_wide.to_csv(stats_path, index=False)
    print(f"1. Model fit stats saved to: {stats_path}")

    # 2. 建立係數摘要 (Coefficient Summary)
    # 轉為寬表格：列為 Building + Variable，欄為不同模型的 Coef
    coef_wide = results_df.pivot(
        index=['Building', 'Variable'], 
        columns='Model', 
        values='Coef'
    )
    
    # 根據原本 MODELS 的順序對欄位進行排序 (若模型有產出的話)
    existing_models = [m['name'] for m in MODELS if m['name'] in coef_wide.columns]
    coef_wide = coef_wide[existing_models]
    
    # 重新命名欄位增加辨識度
    coef_wide.columns = [f"{col}_Coef" for col in coef_wide.columns]
    coef_wide = coef_wide.reset_index()
    
    summary_path = os.path.join(RESULT_FOLDER, "regression_results_summary.csv")
    coef_wide.to_csv(summary_path, index=False)
    print(f"2. Coefficient summary saved to: {summary_path}")

print(f"3. Plots saved to: {PLOT_FOLDER}")
print("\nProcess completed successfully.")