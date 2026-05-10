import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import seaborn as sns
import matplotlib.font_manager as fm

# Font Settings
_CJK_CANDIDATES = ['PingFang TC', 'Heiti TC', 'Noto Sans CJK TC', 'Microsoft JhengHei']
_available = {f.name for f in fm.fontManager.ttflist}
_cjk_font = next((f for f in _CJK_CANDIDATES if f in _available), None)
if _cjk_font: plt.rcParams['font.family'] = _cjk_font
plt.rcParams['axes.unicode_minus'] = False

# Config Paths
WORK_DIR = os.getcwd()
PROCESSED_FOLDER = os.path.join(WORK_DIR, "data", "processed", "alltime")
RESULT_FOLDER = os.path.join(WORK_DIR, "data", "describe")
PLOT_FOLDER = os.path.join(WORK_DIR, "visualization", "temp")

# 確保資料夾存在
os.makedirs(RESULT_FOLDER, exist_ok=True)
os.makedirs(PLOT_FOLDER, exist_ok=True)

# 參數設定
BUILDING_MAP = {"普通": "putong", "新生": "xinsheng", "共同": "gongtong"}
NTU_HOLIDAYS = ["2024-09-17", "2024-10-02", "2024-10-03", "2024-10-04", "2024-10-10", "2024-10-31", "2024-11-22", 
                "2025-02-28", "2025-04-03", "2025-04-04", "2025-05-30", 
                "2025-09-29", "2025-10-06", "2025-10-10", "2025-10-24", "2025-11-21"] 
HOLIDAY_TS = pd.to_datetime(NTU_HOLIDAYS).normalize()
# 使用字串清單，並在讀取時做型態轉換確保匹配
TARGET_PERIODS = [str(i) for i in range(1, 11)]

# Prepare Plot
fig, axes = plt.subplots(3, 1, figsize=(10, 8), sharex=True)
colors = {'Train': '#5b8db8', 'Test': '#e07b54'}

for i, (ch_name, en_name) in enumerate(BUILDING_MAP.items()):
    file_path = os.path.join(PROCESSED_FOLDER, f"{en_name}_final_combined_alltime.csv")
    if not os.path.exists(file_path):
        print(f"找不到檔案: {en_name}")
        continue

    # Load Data
    df = pd.read_csv(file_path)
    df['DateTime'] = pd.to_datetime(df['DateTime'])
    
    # 確保 Period 是字串型態以匹配 TARGET_PERIODS
    df['Period'] = df['Period'].astype(str).str.replace('.0', '', regex=False)
    
    # 標記時期
    df['Type'] = None
    train_idx = (
        ((df['DateTime'] >= '2024-09-02') & (df['DateTime'] <= '2024-12-20')) 
        | ((df['DateTime'] >= '2025-02-24') & (df['DateTime'] <= '2025-06-06'))
    )
    test_idx = (df['DateTime'] >= '2025-09-01') & (df['DateTime'] <= '2025-12-19')
    
    df.loc[train_idx, 'Type'] = 'Train'
    df.loc[test_idx, 'Type'] = 'Test'
    
    # 過濾條件
    is_not_holiday = ~df['DateTime'].dt.normalize().isin(HOLIDAY_TS)
    df_filtered = df[
        df['Type'].notna() & 
        is_not_holiday & 
        df['Period'].isin(TARGET_PERIODS)
    ].copy()

    if df_filtered.empty:
        print(f"警告：{en_name} 過濾後沒有資料。")
        continue

    # 計算日平均用電
    df_filtered['Date'] = df_filtered['DateTime'].dt.normalize()
    daily_df = df_filtered.groupby(['Date', 'Type'], as_index=False)['Electricity'].mean()

    # --- 儲存 CSV ---
    csv_name = f"{en_name}_daily_usage_filtered.csv"
    daily_df.to_csv(os.path.join(RESULT_FOLDER, csv_name), index=False)
    print(f"已儲存資料: {csv_name}")

    # --- 繪圖 ---
    ax = axes[i]
    sns.lineplot(
        data=daily_df, 
        x='Date', 
        y='Electricity', 
        hue='Type', 
        hue_order=['Train', 'Test'],
        palette=colors, 
        ax=ax, 
        linewidth=1.2
    )
    
    _title = ch_name if _cjk_font else en_name
    ax.set_title(f"{_title} 日均用電變化 (第 {TARGET_PERIODS[0]}-{TARGET_PERIODS[-1]} 節)", 
                 fontsize=12, fontweight='bold')
    ax.set_ylabel("kWh")
    ax.grid(True, linestyle=':', alpha=0.6)
    ax.legend(title='時期', loc='upper right')

# 設定 X 軸
axes[-1].xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
axes[-1].xaxis.set_major_locator(mdates.MonthLocator(interval=1))
plt.xticks(rotation=45)
plt.tight_layout()

# --- 儲存圖片 ---
plot_path = os.path.join(PLOT_FOLDER, "buildings_daily_usage_comparison.png")
plt.savefig(plot_path, dpi=200, bbox_inches='tight')
print(f"已儲存圖表: {plot_path}")

plt.show()