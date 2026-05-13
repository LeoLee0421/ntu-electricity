import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import matplotlib.gridspec as gridspec
import matplotlib.font_manager as fm

# ── Font ───────────────────────────────────────────────────────────────────────
_CJK_CANDIDATES = ['PingFang TC', 'Heiti TC', 'Noto Sans CJK TC', 'Microsoft JhengHei']
_available  = {f.name for f in fm.fontManager.ttflist}
_cjk_font   = next((f for f in _CJK_CANDIDATES if f in _available), None)
if _cjk_font:
    plt.rcParams['font.family'] = _cjk_font
plt.rcParams['axes.unicode_minus'] = False

# ── Paths ──────────────────────────────────────────────────────────────────────
WORK_DIR      = os.getcwd()
RESULT_FOLDER = os.path.join(WORK_DIR, "data", "results")
PLOT_FOLDER   = os.path.join(WORK_DIR, "visualization")
os.makedirs(PLOT_FOLDER, exist_ok=True)

BUILDING_MAP    = {"普通": "putong", "新生": "xinsheng", "共同": "gongtong"}
PALETTE         = {'actual': '#2c2c2c', 'pred': '#6ab187', 'residual': '#e07b54'}
SCENARIO_COLORS = ['#5b8db8', '#4aab82', '#e07b54', '#a855f7']  # 最多4間，多備一色

# 各棟 BigC 上限（與 run_model.py 一致）
BIGC_MAX = {'putong': 4, 'xinsheng': 3, 'gongtong': 3}


# ════════════════════════════════════════════════════════════════
# 圖 1：模型效能比較（3棟並列）
#   - 左欄：In-sample vs Out-of-sample R² bar chart
#   - 右欄：In-sample vs Out-of-sample RMSE bar chart
# ════════════════════════════════════════════════════════════════
def plot_model_performance():
    records = []
    for ch, en in BUILDING_MAP.items():
        path = os.path.join(RESULT_FOLDER, f"{en}_model_metrics.csv")
        if not os.path.exists(path): continue
        row       = pd.read_csv(path).iloc[0]
        row['ch'] = ch
        records.append(row)
    if not records:
        print("[SKIP] 沒有找到 metrics CSV，請先執行 run_model.py")
        return

    df_m   = pd.DataFrame(records)
    x      = np.arange(len(df_m))
    width  = 0.35
    labels = df_m['ch'].tolist() if _cjk_font else df_m['Building'].tolist()

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.5))
    fig.suptitle("模型效能概覽（訓練 vs 測試）", fontsize=14, fontweight='bold')

    # R²
    b1 = ax1.bar(x - width/2, df_m['R2_in'],  width, label='In-sample',  color='#5b8db8', alpha=0.85)
    b2 = ax1.bar(x + width/2, df_m['R2_oos'], width, label='Out-of-sample', color='#6ab187', alpha=0.85)
    ax1.set_xticks(x); ax1.set_xticklabels(labels, fontsize=12)
    ax1.set_ylabel("R²"); ax1.set_ylim(0, 1.05)
    ax1.axhline(0.9, color='grey', linestyle=':', alpha=0.6, label='R²=0.9 參考線')
    ax1.legend(); ax1.grid(axis='y', linestyle=':', alpha=0.5)
    for bar in [*b1, *b2]:
        ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                 f"{bar.get_height():.3f}", ha='center', va='bottom', fontsize=9)

    # RMSE
    b3 = ax2.bar(x - width/2, df_m['RMSE_in'],  width, label='In-sample',  color='#5b8db8', alpha=0.85)
    b4 = ax2.bar(x + width/2, df_m['RMSE_oos'], width, label='Out-of-sample', color='#6ab187', alpha=0.85)
    ax2.set_xticks(x); ax2.set_xticklabels(labels, fontsize=12)
    ax2.set_ylabel("RMSE (kWh)"); ax2.legend()
    ax2.grid(axis='y', linestyle=':', alpha=0.5)
    for bar in [*b3, *b4]:
        ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.3,
                 f"{bar.get_height():.2f}", ha='center', va='bottom', fontsize=9)

    plt.tight_layout()
    out = os.path.join(PLOT_FOLDER, "01_model_performance.png")
    plt.savefig(out, dpi=200, bbox_inches='tight')
    plt.close()
    print(f"[OK] 圖1 儲存：{out}")


# ════════════════════════════════════════════════════════════════
# 圖 2：各棟 Actual vs Predicted 時間序列 + Scatter（3列）
# ════════════════════════════════════════════════════════════════
def plot_actual_vs_predicted():
    fig, axes = plt.subplots(3, 2, figsize=(16, 12),
                             gridspec_kw={'width_ratios': [3, 1]})
    fig.suptitle("各棟建築：實際 vs 預測用電（測試期）", fontsize=14, fontweight='bold')

    for row_i, (ch, en) in enumerate(BUILDING_MAP.items()):
        path = os.path.join(RESULT_FOLDER, f"{en}_predictions.csv")
        if not os.path.exists(path): continue
        df = pd.read_csv(path)
        df['DateTime'] = pd.to_datetime(df['DateTime'])

        # 日均平滑
        daily = df.set_index('DateTime').resample('D').mean().dropna().reset_index()
        label = ch if _cjk_font else en

        # ── 左：時間序列 ──
        ax_ts = axes[row_i][0]
        ax_ts.plot(daily['DateTime'], daily['Actual'],    color=PALETTE['actual'],
                   label='實際', alpha=0.75, linewidth=1.2)
        ax_ts.plot(daily['DateTime'], daily['Predicted'], color=PALETTE['pred'],
                   label='預測', linestyle='--', linewidth=1.2)
        ax_ts.fill_between(daily['DateTime'],
                            daily['Actual'], daily['Predicted'],
                            alpha=0.12, color=PALETTE['pred'])
        ax_ts.set_title(f"{label}", fontsize=11, fontweight='bold')
        ax_ts.set_ylabel("kWh")
        ax_ts.legend(loc='upper right', fontsize=9)
        ax_ts.grid(True, linestyle=':', alpha=0.5)
        ax_ts.xaxis.set_major_formatter(mdates.DateFormatter('%m-%d'))
        plt.setp(ax_ts.xaxis.get_majorticklabels(), rotation=30, ha='right')

        # ── 右：Scatter（Actual vs Predicted）──
        ax_sc = axes[row_i][1]
        ax_sc.scatter(df['Actual'], df['Predicted'],
                      alpha=0.15, s=8, color=PALETTE['pred'])
        lim = [min(df['Actual'].min(), df['Predicted'].min()),
               max(df['Actual'].max(), df['Predicted'].max())]
        ax_sc.plot(lim, lim, 'r--', linewidth=1, label='45° line')
        ax_sc.set_xlabel("實際 (kWh)", fontsize=9)
        ax_sc.set_ylabel("預測 (kWh)", fontsize=9)
        ax_sc.legend(fontsize=8)
        ax_sc.grid(True, linestyle=':', alpha=0.5)

    plt.tight_layout()
    out = os.path.join(PLOT_FOLDER, "02_actual_vs_predicted.png")
    plt.savefig(out, dpi=200, bbox_inches='tight')
    plt.close()
    print(f"[OK] 圖2 儲存：{out}")


# ════════════════════════════════════════════════════════════════
# 圖 3：情境模擬——少開 N 間大教室可省多少電
#   - 共用 Y 軸（sharey=True）
#   - 各棟上限由 BIGC_MAX 決定
#   - 普通館標注資料僅含 11–12 月
# ════════════════════════════════════════════════════════════════
def plot_scenario_savings():
    all_data = {}
    for ch, en in BUILDING_MAP.items():
        path = os.path.join(RESULT_FOLDER, f"{en}_scenario_savings.csv")
        if not os.path.exists(path): continue
        df = pd.read_csv(path)
        df['ch'] = ch
        # 依 BIGC_MAX 過濾，排除超過實際間數的模擬列
        df = df[df['Reduce_BigC_N'] <= BIGC_MAX[en]].copy()
        all_data[en] = df

    if not all_data:
        print("[SKIP] 找不到 scenario CSV")
        return

    # 共用 Y 軸：取全局最大值
    global_max = max(df['Total_Savings_kWh'].max() for df in all_data.values())

    fig, axes = plt.subplots(1, 3, figsize=(15, 7), sharey=True)  # sharey=True
    fig.suptitle("情境模擬：少開大教室的節電效益", fontsize=14, fontweight='bold')

    for col_i, (en, df) in enumerate(all_data.items()):
        ax        = axes[col_i]
        ch        = df['ch'].iloc[0]
        ns        = df['Reduce_BigC_N'].tolist()
        savings   = df['Total_Savings_kWh'].tolist()
        test_days = int(df['Test_Days'].iloc[0])
        is_partial = (en == 'putong')

        bars = ax.bar(ns, savings,
                      color=SCENARIO_COLORS[:len(ns)],
                      width=0.5, alpha=0.85, edgecolor='white')

        for bar, kwh in zip(bars, savings):
            ntd = kwh * 3.5
            ax.text(bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + global_max * 0.02,
                    f"{kwh:,.0f} kWh\n≈ NT${ntd:,.0f}",
                    ha='center', va='bottom', fontsize=9, fontweight='bold')

        # 建築名稱 + 副標題合成兩行標題
        # 不用 ax.text 浮動，改用 set_title 的多行字串，讓 tight_layout 正確計算空間
        subtitle = (
            "資料僅含 11\u201312 月，不宜跨棟比較"
            if is_partial else
            f"完整學期，共 {test_days} 天"
        )
        building_label = ch if _cjk_font else en
        ax.set_title(
            f"{building_label}\n{subtitle}",
            fontsize=11, fontweight='bold',
            color='#e07b54' if is_partial else 'black',
            pad=8,
        )
        ax.set_xlabel("減少大教室使用間數 (間)")
        if col_i == 0:
            ax.set_ylabel("測試期總省電量 (kWh)")
        ax.set_xticks(ns)
        ax.set_xticklabels([f"少開 {n} 間" for n in ns])
        ax.set_ylim(0, global_max * 1.25)
        ax.grid(axis='y', linestyle=':', alpha=0.5)
        ax.spines[['top', 'right']].set_visible(False)

    plt.tight_layout(rect=[0, 0, 1, 0.93])  # 頂部留空給 suptitle
    out = os.path.join(PLOT_FOLDER, "03_scenario_savings.png")
    plt.savefig(out, dpi=200, bbox_inches='tight')
    plt.close()
    print(f"[OK] 圖3 儲存：{out}")


# ════════════════════════════════════════════════════════════════
# 圖 4：BigC、SmallC 及交乘項係數（各棟 + 95% CI）
# ════════════════════════════════════════════════════════════════
def plot_classroom_coefficients():
    TARGET_VARS = {
        'BigC'           : '大教室數量\n(BigC)',
        'SmallC'         : '小教室數量\n(SmallC)',
        'Temp_c_x_BigC'  : '溫度×大教室\n(Temp_c × BigC)',
        'Temp_c_x_SmallC': '溫度×小教室\n(Temp_c × SmallC)',
    }
    COLORS_BUILD = {'putong': '#5b8db8', 'xinsheng': '#4aab82', 'gongtong': '#e07b54'}

    build_coefs = {}
    for ch, en in BUILDING_MAP.items():
        path = os.path.join(RESULT_FOLDER, f"{en}_coefficients.csv")
        if not os.path.exists(path): continue
        coef_df = pd.read_csv(path).set_index('Variable')
        build_coefs[en] = {'ch': ch, 'data': coef_df}

    if not build_coefs:
        print("[SKIP] 沒有係數資料")
        return

    n_vars   = len(TARGET_VARS)
    n_builds = len(build_coefs)
    fig, axes = plt.subplots(1, n_vars, figsize=(14, 5), sharey=False)
    fig.suptitle("教室使用係數與交乘項（各棟，含 95% CI）",
                 fontsize=13, fontweight='bold')

    for ax, (var_key, var_label) in zip(axes, TARGET_VARS.items()):
        y_positions = np.arange(n_builds)
        en_list     = list(build_coefs.keys())

        for j, en in enumerate(en_list):
            row = build_coefs[en]['data']
            if var_key not in row.index:
                continue
            coef  = row.loc[var_key, 'Coef']
            se    = row.loc[var_key, 'StdErr']
            pval  = row.loc[var_key, 'P_value']
            ci95  = 1.96 * se
            color = COLORS_BUILD[en]

            ax.barh(j, coef, xerr=ci95,
                    color=color, alpha=0.80, capsize=5, height=0.5,
                    error_kw={'elinewidth': 1.5, 'ecolor': 'dimgray'})

            sig = ('***' if pval < 0.001 else
                   '**'  if pval < 0.01  else
                   '*'   if pval < 0.05  else 'n.s.')
            sig_color = '#2c7a2c' if sig != 'n.s.' else 'grey'
            ax.text(coef + ci95 + abs(coef) * 0.05,
                    j, sig, va='center', ha='left',
                    fontsize=10, color=sig_color, fontweight='bold')

        ax.axvline(0, color='black', linewidth=0.9, linestyle='--', alpha=0.6)
        ax.set_yticks(y_positions)
        ax.set_yticklabels(
            [build_coefs[en]['ch'] if _cjk_font else en for en in en_list],
            fontsize=10
        )
        ax.set_title(var_label, fontsize=10, fontweight='bold', pad=8)
        ax.set_xlabel("係數 (kWh)", fontsize=9)
        ax.grid(axis='x', linestyle=':', alpha=0.5)
        ax.spines[['top', 'right']].set_visible(False)

    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor=COLORS_BUILD[en], alpha=0.8,
              label=build_coefs[en]['ch'] if _cjk_font else en)
        for en in build_coefs
    ]
    fig.legend(handles=legend_elements,
               loc='lower center', ncol=n_builds,
               bbox_to_anchor=(0.5, -0.04),
               frameon=False, fontsize=10)

    plt.tight_layout(rect=[0, 0.04, 1, 1])
    out = os.path.join(PLOT_FOLDER, "04_classroom_coefficients.png")
    plt.savefig(out, dpi=200, bbox_inches='tight')
    plt.close()
    print(f"[OK] 圖4 儲存：{out}")


# ── 執行全部 ──────────────────────────────────────────────────
if __name__ == '__main__':
    plot_model_performance()
    plot_actual_vs_predicted()
    plot_scenario_savings()
    plot_classroom_coefficients()
    print("\n[完成] 所有視覺化圖表已輸出至:", PLOT_FOLDER)