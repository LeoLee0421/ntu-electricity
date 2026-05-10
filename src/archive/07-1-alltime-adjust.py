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
    # No CJK font found — replace Chinese building names with English in plot titles
    print("Warning: No CJK font found. Chinese characters in plot titles will be replaced with English names.")
plt.rcParams['axes.unicode_minus'] = False   # prevent minus sign rendering issue
 
 

# Set paths ------------------------------------------------------------------------------------
WORK_DIR = os.getcwd()
PROCESSED_FOLDER = os.path.join(WORK_DIR, "data", "processed", "alltime")
RESULT_FOLDER = os.path.join(WORK_DIR, "data", "results", "adjust_alltime")
PLOT_FOLDER = os.path.join(WORK_DIR, "visualization", "predict_outcome_alltime")
os.makedirs(RESULT_FOLDER, exist_ok=True)
os.makedirs(PLOT_FOLDER, exist_ok=True)

# 建築對應（中文 → 英文）
BUILDING_MAP = {
    "普通": "putong",
    #"綜合": "zonghe",
    #"博雅": "boya",
    "新生": "xinsheng",
    "共同": "gongtong"
}

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

    # Create dummies; drop_first=True avoids perfect multicollinearity
    month_dummies   = pd.get_dummies(df['month'],   prefix='month',   drop_first=True).astype(int)
    weekday_dummies = pd.get_dummies(df['weekday'], prefix='weekday', drop_first=True).astype(int)
    hour_dummies    = pd.get_dummies(df['hour'],    prefix='hour',    drop_first=True).astype(int)

    X_base = df[BASE_FEATURES].reset_index(drop=True)
    X_all  = pd.concat(
        [X_base,
         month_dummies.reset_index(drop=True),
         weekday_dummies.reset_index(drop=True),
         hour_dummies.reset_index(drop=True)],
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

# Main loop ------------------------------------------------------------------------------------
results_list = []   # coefficient-level rows
pred_dict    = {}   # { (en_name, model_name): prediction_df }

for ch_name, en_name in BUILDING_MAP.items():

    # --- Load data ---
    file_path = os.path.join(PROCESSED_FOLDER, f"{en_name}_final_combined_alltime.csv")
    if not os.path.exists(file_path):
        print(f"File not found, skipping: {file_path}")
        continue

    df = pd.read_csv(file_path)
    df['DateTime'] = pd.to_datetime(df['DateTime'])

    # --- Train / test split ---
    train_mask = (
        ((df['DateTime'] >= train_period_1[0]) & (df['DateTime'] <= train_period_1[1])) |
        ((df['DateTime'] >= train_period_2[0]) & (df['DateTime'] <= train_period_2[1]))
    )
    test_mask = (df['DateTime'] >= test_period[0]) & (df['DateTime'] <= test_period[1])

    df_train = df[train_mask].copy().reset_index(drop=True)
    df_test  = df[test_mask].copy().reset_index(drop=True)

    if df_train.empty or df_test.empty:
        print(f"Warning: Insufficient data for {en_name}, skipping.")
        continue

    y_train = df_train["Electricity"]
    y_test  = df_test["Electricity"]

    # --- Run each model ---
    for model_cfg in MODELS:
        model_name = model_cfg['name']
        build_fn   = model_cfg['build_fn']

        try:
            X_train, feature_names = build_fn(df_train)
            res = sm.OLS(y_train, X_train).fit()

            # --- Save coefficients ---
            for feat in feature_names:
                if feat in res.params.index:
                    results_list.append({
                        'Building': en_name,
                        'Model':    model_name,
                        'Variable': feat,
                        'Coef':     res.params[feat],
                        'StdErr':   res.bse[feat],
                        'tValue':   res.tvalues[feat],
                        'pValue':   res.pvalues[feat],
                        'R2':       res.rsquared,
                        'Adj_R2':   res.rsquared_adj,
                        'N_train':  int(res.nobs),
                    })

            # --- Predict on test set ---
            X_test, _ = build_fn(df_test)
            X_test = X_test.reindex(columns=X_train.columns, fill_value=0)
            y_pred = res.predict(X_test)

            prediction_df = pd.DataFrame({
                'DateTime':              df_test['DateTime'],
                'Actual_Electricity':    y_test.values,
                'Predicted_Electricity': y_pred.values,
                'Residual':              (y_test - y_pred).values,
            })

            pred_filename = f"{en_name}_{model_name}_test_prediction.csv"
            prediction_df.to_csv(os.path.join(RESULT_FOLDER, pred_filename), index=False)
            print(f"Saved prediction CSV: {pred_filename}")

            pred_dict[(en_name, model_name)] = prediction_df

        except Exception as e:
            print(f"Error running {model_name} for {en_name}: {e}")
            traceback.print_exc()

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

    # ── Figure 1: Time-series actual vs predicted (daily mean, one subplot per model) ──────────
    fig, axes = plt.subplots(n_models, 1, figsize=(16, 4 * n_models), sharex=False)
    if n_models == 1:
        axes = [axes]

    fig.suptitle(
        f"{ch_name} ({en_name})  —  Test Period: Actual vs Predicted Electricity (Daily Mean)",
        fontsize=13, fontweight='bold', y=1.01
    )

    for ax, (model_cfg, pdf) in zip(axes, model_preds):
        model_name  = model_cfg['name']
        model_label = model_cfg['label']
        color       = COLORS.get(model_name, '#888888')

        # Resample to daily mean — hourly data is too dense to read clearly
        pdf_plot = pdf.set_index('DateTime').resample('D').mean().reset_index()

        rmse = np.sqrt((pdf['Residual'] ** 2).mean())
        r2   = np.corrcoef(pdf['Actual_Electricity'], pdf['Predicted_Electricity'])[0, 1] ** 2

        ax.plot(pdf_plot['DateTime'], pdf_plot['Actual_Electricity'],
                color=COLORS['actual'], linewidth=1.2, label='Actual', alpha=0.9)
        ax.plot(pdf_plot['DateTime'], pdf_plot['Predicted_Electricity'],
                color=color, linewidth=1.2, linestyle='--', label='Predicted', alpha=0.9)

        ax.set_title(f"{model_label}   |   RMSE = {rmse:.2f}   Out-of-sample R² = {r2:.3f}",
                     fontsize=10, pad=5)
        ax.set_ylabel("Electricity (kWh)", fontsize=9)
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
        ax.xaxis.set_major_locator(mdates.MonthLocator())
        ax.tick_params(axis='x', rotation=30, labelsize=8)
        ax.legend(fontsize=9, loc='upper right')
        ax.grid(True, linestyle=':', alpha=0.5)

    fig.tight_layout()
    ts_path = os.path.join(PLOT_FOLDER, f"{en_name}_timeseries_actual_vs_predicted.png")
    fig.savefig(ts_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"Saved time-series plot: {ts_path}")

    # ── Figure 2: Scatter actual vs predicted (one subplot per model) ──────────────────────────
    fig2, axes2 = plt.subplots(1, n_models, figsize=(5 * n_models, 5))
    if n_models == 1:
        axes2 = [axes2]

    fig2.suptitle(f"{ch_name} ({en_name})  —  Scatter: Actual vs Predicted",
                  fontsize=13, fontweight='bold')

    for ax2, (model_cfg, pdf) in zip(axes2, model_preds):
        model_name  = model_cfg['name']
        model_label = model_cfg['label']
        color       = COLORS.get(model_name, '#888888')

        actual    = pdf['Actual_Electricity']
        predicted = pdf['Predicted_Electricity']
        rmse = np.sqrt((pdf['Residual'] ** 2).mean())
        r2   = np.corrcoef(actual, predicted)[0, 1] ** 2

        ax2.scatter(actual, predicted, alpha=0.15, s=5, color=color)

        # 45-degree perfect-fit reference line
        lim_min = min(actual.min(), predicted.min())
        lim_max = max(actual.max(), predicted.max())
        ax2.plot([lim_min, lim_max], [lim_min, lim_max],
                 color='#2c2c2c', linewidth=1, linestyle='--', label='Perfect fit')

        ax2.set_title(f"{model_label}\nRMSE = {rmse:.2f}   R² = {r2:.3f}", fontsize=9)
        ax2.set_xlabel("Actual (kWh)", fontsize=9)
        ax2.set_ylabel("Predicted (kWh)", fontsize=9)
        ax2.legend(fontsize=8)
        ax2.grid(True, linestyle=':', alpha=0.5)

    fig2.tight_layout()
    sc_path = os.path.join(PLOT_FOLDER, f"{en_name}_scatter_actual_vs_predicted.png")
    fig2.savefig(sc_path, dpi=150, bbox_inches='tight')
    plt.close(fig2)
    print(f"Saved scatter plot:      {sc_path}")

# Final summary --------------------------------------------------------------------------------
results_df = pd.DataFrame(results_list)

model_stats = (
    results_df[['Building', 'Model', 'R2', 'Adj_R2', 'N_train']]
    .drop_duplicates()
)

coef_wide = results_df.pivot_table(
    index=['Building', 'Variable'],
    columns='Model',
    values=['Coef', 'StdErr', 'tValue', 'pValue'],
    aggfunc='first'
)
coef_wide.columns = [f"{model}_{stat}" for stat, model in coef_wide.columns]
coef_wide = coef_wide.reset_index()

model_names  = [m['name'] for m in MODELS]
ordered_cols = ['Building', 'Variable']
for mn in model_names:
    for stat in ['Coef']:
        col = f"{mn}_{stat}"
        if col in coef_wide.columns:
            ordered_cols.append(col)
coef_wide = coef_wide[ordered_cols]

summary_path = os.path.join(RESULT_FOLDER, "regression_results_summary.csv")
coef_wide.to_csv(summary_path, index=False)

stats_path = os.path.join(RESULT_FOLDER, "regression_model_stats.csv")
model_stats_wide = model_stats.pivot_table(
    index='Building',
    columns='Model',
    values=['R2', 'Adj_R2', 'N_train'],
    aggfunc='first'
)
model_stats_wide.columns = [f"{model}_{stat}" for stat, model in model_stats_wide.columns]
model_stats_wide = model_stats_wide.reset_index()
model_stats_wide.to_csv(stats_path, index=False)

print("\nProcess completed successfully.")
print(f"1. Coefficient summary (wide format) saved to: {summary_path}")
print(f"2. Model fit stats (R2, Adj_R2, N) saved to:   {stats_path}")
print(f"3. Plots saved to:                              {PLOT_FOLDER}")