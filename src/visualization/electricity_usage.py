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
if _cjk_font:
    plt.rcParams['font.family'] = _cjk_font
plt.rcParams['axes.unicode_minus'] = False

# Config Paths
WORK_DIR = os.getcwd()
PROCESSED_FOLDER = os.path.join(WORK_DIR, "data", "processed", "alltime")
RESULT_FOLDER = os.path.join(WORK_DIR, "data", "describe")
PLOT_FOLDER = os.path.join(WORK_DIR, "visualization", "temp")

os.makedirs(RESULT_FOLDER, exist_ok=True)
os.makedirs(PLOT_FOLDER, exist_ok=True)

# 參數設定
BUILDING_MAP = {"普通": "putong", "新生": "xinsheng", "共同": "gongtong"}
NTU_HOLIDAYS = [
    "2024-09-17", "2024-10-02", "2024-10-03", "2024-10-04", "2024-10-10",
    "2024-10-31", "2024-11-22",
    "2025-02-28", "2025-04-03", "2025-04-04", "2025-05-30",
    "2025-09-29", "2025-10-06", "2025-10-10", "2025-10-24", "2025-11-21"
]
HOLIDAY_TS = pd.to_datetime(NTU_HOLIDAYS).normalize()
TARGET_PERIODS = [str(i) for i in range(1, 11)]

# ── 學期定義 ──────────────────────────────────────────────────
SEMESTERS = {
    "113 上學期\n(2024/09–12)": ("2024-09-02", "2024-12-20"),
    "114 上學期\n(2025/09–12)": ("2025-09-01", "2025-12-19"),
}
SEM_KEYS  = list(SEMESTERS.keys())
SEM_COLOR = {"113 上學期\n(2024/09–12)": "#3f1163",
             "114 上學期\n(2025/09–12)": "#e6922b"}

# ── 建立 2×3 Figure（列=學期, 欄=建築），共用每欄的 Y 軸 ──────
fig, axes = plt.subplots(
    2, 3,
    figsize=(14, 7),
    sharey='col',   # 同一欄（同棟建築）共用 Y 軸
    sharex=False    # 各格 X 軸獨立
)
fig.suptitle("各棟建築各學期日均用電量（第 1–10 節，排除假日）",
             fontsize=14, fontweight='bold', y=1.01)

for col_j, (ch_name, en_name) in enumerate(BUILDING_MAP.items()):
    file_path = os.path.join(PROCESSED_FOLDER, f"{en_name}_final_combined_alltime.csv")
    if not os.path.exists(file_path):
        print(f"找不到檔案: {en_name}")
        continue

    # ── 讀取資料 ──
    df = pd.read_csv(file_path)
    df['DateTime'] = pd.to_datetime(df['DateTime'])
    df['Period']   = df['Period'].astype(str).str.replace('.0', '', regex=False)

    is_not_holiday = ~df['DateTime'].dt.normalize().isin(HOLIDAY_TS)
    df = df[is_not_holiday & df['Period'].isin(TARGET_PERIODS)].copy()

    for row_i, sem_label in enumerate(SEM_KEYS):
        start, end = SEMESTERS[sem_label]
        ax = axes[row_i][col_j]

        sem_df = df[(df['DateTime'] >= start) & (df['DateTime'] <= end)].copy()

        if sem_df.empty:
            ax.text(0.5, 0.5, '無資料', ha='center', va='center',
                    transform=ax.transAxes, color='grey')
            ax.set_title(f"{ch_name if _cjk_font else en_name} | {sem_label.split(chr(10))[0]}",
                         fontsize=9)
            continue

        # 日平均
        sem_df['Date'] = sem_df['DateTime'].dt.normalize()
        daily_df = sem_df.groupby('Date', as_index=False)['Electricity'].mean()

        # 儲存 CSV（每棟×每學期）
        sem_tag = ["train_s1", "test"][row_i]
        csv_name = f"{en_name}_{sem_tag}_daily_usage.csv"
        daily_df.to_csv(os.path.join(RESULT_FOLDER, csv_name), index=False)

        # 繪圖
        color = SEM_COLOR[sem_label]
        ax.plot(daily_df['Date'], daily_df['Electricity'],
                color=color, linewidth=1.2, alpha=0.85)
        ax.fill_between(daily_df['Date'], daily_df['Electricity'],
                        color=color, alpha=0.12)

        # X 軸格式（月份）
        ax.xaxis.set_major_locator(mdates.MonthLocator(interval=1))
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%m月'))
        plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha='right', fontsize=8)

        ax.grid(True, linestyle=':', alpha=0.5)
        ax.set_xlabel("")

        # 標題：只在第一列顯示建築名，只在第一欄顯示學期
        if row_i == 0:
            building_label = ch_name if _cjk_font else en_name
            ax.set_title(building_label, fontsize=10, fontweight='bold', color=color)
        if col_j == 0:
            ax.set_ylabel(f"{sem_label}\nkWh", fontsize=10, fontweight='bold')
        else:
            ax.set_ylabel("")

plt.tight_layout()

plot_path = os.path.join(PLOT_FOLDER, "buildings_semester_2x3_comparison.png")
plt.savefig(plot_path, dpi=200, bbox_inches='tight', transparent=True)
print(f"已儲存圖表: {plot_path}")
plt.show()