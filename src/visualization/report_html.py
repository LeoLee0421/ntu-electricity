"""
generate_report.py
------------------
讀取 run_model.py 產出的 CSV，生成一份自給自足的互動式 HTML 報告。
執行方式：python generate_report.py
輸出：visualization/electricity_report.html
"""

import os, json
import pandas as pd
import numpy as np

# ── Paths ──────────────────────────────────────────────────────────────────────
WORK_DIR      = os.getcwd()
RESULT_FOLDER = os.path.join(WORK_DIR, "data", "results")
PLOT_FOLDER   = os.path.join(WORK_DIR, "visualization")
os.makedirs(PLOT_FOLDER, exist_ok=True)

BUILDING_MAP = {"普通": "putong", "新生": "xinsheng", "共同": "gongtong"}
ELECTRICITY_PRICE_NTD = 3.5

# ── Load all data into a dict ──────────────────────────────────────────────────
data = {}
for ch, en in BUILDING_MAP.items():
    def _read(name):
        p = os.path.join(RESULT_FOLDER, f"{en}_{name}.csv")
        return pd.read_csv(p) if os.path.exists(p) else pd.DataFrame()

    metrics_df  = _read("model_metrics")
    pred_df     = _read("predictions")
    coef_df     = _read("coefficients")
    scenario_df = _read("scenario_savings")

    # daily.csv 不存在時，直接從 predictions.csv 聚合日均值
    if not pred_df.empty:
        _tmp = pred_df.copy()
        _tmp['Date'] = pd.to_datetime(_tmp['DateTime']).dt.date.astype(str)
        daily_df = _tmp.groupby('Date')[['Actual','Predicted','Residual']].mean().round(4).reset_index()
    else:
        daily_df = pd.DataFrame()

    if metrics_df.empty:
        print(f"[WARN] {en}: 找不到 metrics CSV，跳過")
        continue

    m = metrics_df.iloc[0]

    # 普通館旗標：測試資料不完整（僅 11–12 月）
    is_partial = (en == 'putong')

    # 殘差直方圖 bin（30 bins）
    residuals = pred_df['Residual'].dropna().tolist() if not pred_df.empty else []
    if residuals:
        hist_counts, hist_edges = np.histogram(residuals, bins=30)
        hist_data = {
            'counts': hist_counts.tolist(),
            'edges' : hist_edges.tolist(),
        }
    else:
        hist_data = {'counts': [], 'edges': []}

    # 係數：排除 dummy，保留核心變數
    if not coef_df.empty:
        core_coefs = coef_df[~coef_df['Variable'].str.startswith(('month_', 'hour_'))].copy()
        core_coefs['sig'] = core_coefs['P_value'].apply(
            lambda p: '***' if p < 0.001 else ('**' if p < 0.01 else ('*' if p < 0.05 else 'n.s.')))
    else:
        core_coefs = pd.DataFrame()

    # scenario 欄位名稱對齊 run_model.py 的大寫輸出
    scenario_records = []
    if not scenario_df.empty:
        for _, row in scenario_df.iterrows():
            scenario_records.append({
                'reduce_n'    : int(row['Reduce_BigC_N']),
                'total_kwh'   : float(row['Total_Savings_kWh']),
                'avg_per_day' : float(row['Avg_Savings_per_Day']),
                'test_days'   : int(row['Test_Days']),
                'total_ntd'   : round(float(row['Total_Savings_kWh']) * ELECTRICITY_PRICE_NTD, 0),
            })

    # BigC 上限寫死（普通=4, 新生=3, 共同=3）
    BIGC_MAX = {'putong': 4, 'xinsheng': 3, 'gongtong': 3}
    bigc_max = BIGC_MAX.get(en, 3)

    data[en] = {
        'ch'        : ch,
        'is_partial': is_partial,   # ← 普通館警示旗標
        'bigc_max'  : bigc_max,     # ← 動態情境上限
        'metrics'   : {
            'r2_in'   : float(m['R2_in']),
            'rmse_in' : float(m['RMSE_in']),
            'r2_oos'  : float(m['R2_oos']),
            'rmse_oos': float(m['RMSE_oos']),
            'n_train' : int(m['N_train']),
            'n_test'  : int(m['N_test']),
        },
        'daily'   : daily_df.to_dict(orient='list') if not daily_df.empty else {},
        'scatter' : {
            'actual'   : pred_df['Actual'].round(3).tolist()    if not pred_df.empty else [],
            'predicted': pred_df['Predicted'].round(3).tolist() if not pred_df.empty else [],
        },
        'hist'    : hist_data,
        'coefs'   : core_coefs.to_dict(orient='records') if not core_coefs.empty else [],
        'scenario': scenario_records,
    }

data_json = json.dumps(data, ensure_ascii=False)

# ── HTML Template ──────────────────────────────────────────────────────────────
HTML = f"""<!DOCTYPE html>
<html lang="zh-TW">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>校舍用電分析報告</title>
<script src="https://cdnjs.cloudflare.com/ajax/libs/Chart.js/4.4.1/chart.umd.min.js"></script>
<style>
  /* ── Design System ── */
  :root {{
    --bg       : #0d0a14;
    --surface  : #1a1427;
    --surface2 : #231933;
    --border   : #3f2060;
    --accent   : #e6922b;
    --accent2  : #9d94c0;
    --accent3  : #fedfb2;
    --text     : #f0ece8;
    --muted    : #9d94c0;
    --danger   : #f87171;
    --font-head: 'Georgia', 'Noto Serif TC', serif;
    --font-body: 'Helvetica Neue', 'PingFang TC', sans-serif;
  }}
  * {{ box-sizing: border-box; margin: 0; padding: 0; }}
  html {{ scroll-behavior: smooth; }}
  body {{
    background: var(--bg);
    color: var(--text);
    font-family: var(--font-body);
    font-size: 14px;
    line-height: 1.6;
  }}

  /* ── Header ── */
  .hero {{
    background: linear-gradient(135deg, #0d0a14 0%, #1a1427 50%, #3f1163 100%);
    border-bottom: 1px solid var(--border);
    padding: 48px 40px 36px;
    position: relative;
    overflow: hidden;
  }}
  .hero::before {{
    content: '';
    position: absolute; inset: 0;
    background: radial-gradient(ellipse at 70% 50%, rgba(63,17,99,.2) 0%, transparent 60%);
  }}
  .hero-label {{
    font-size: 11px; letter-spacing: 3px; text-transform: uppercase;
    color: var(--accent); margin-bottom: 12px; font-weight: 600;
  }}
  .hero h1 {{
    font-family: var(--font-head);
    font-size: clamp(24px, 4vw, 40px);
    font-weight: 700; line-height: 1.2;
    color: #fff; margin-bottom: 10px;
  }}
  .hero p {{ color: var(--muted); font-size: 14px; max-width: 600px; }}
  .hero-badges {{ display: flex; gap: 10px; margin-top: 20px; flex-wrap: wrap; }}
  .badge {{
    padding: 4px 12px; border-radius: 99px; font-size: 11px; font-weight: 600;
    letter-spacing: .5px; border: 1px solid;
  }}
  .badge-green {{ background: rgba(230,146,43,.1); color: var(--accent); border-color: rgba(230,146,43,.3); }}
  .badge-blue  {{ background: rgba(157,148,192,.1); color: var(--accent2); border-color: rgba(157,148,192,.3); }}
  .badge-org   {{ background: rgba(254,223,178,.1);  color: var(--accent3); border-color: rgba(254,223,178,.3); }}

  /* ── Nav ── */
  nav {{
    position: sticky; top: 0; z-index: 100;
    background: rgba(15,17,23,.92); backdrop-filter: blur(12px);
    border-bottom: 1px solid var(--border);
    display: flex; gap: 0; overflow-x: auto;
  }}
  nav a {{
    padding: 14px 20px; font-size: 13px; font-weight: 500;
    color: var(--muted); text-decoration: none; white-space: nowrap;
    border-bottom: 2px solid transparent; transition: all .2s;
  }}
  nav a:hover, nav a.active {{ color: var(--text); border-color: var(--accent); }}

  /* ── Layout ── */
  .container {{ max-width: 1200px; margin: 0 auto; padding: 0 24px; }}
  section {{ padding: 48px 0; border-bottom: 1px solid var(--border); }}
  .section-title {{
    font-family: var(--font-head);
    font-size: 22px; font-weight: 700; color: #fff;
    margin-bottom: 6px;
  }}
  .section-sub {{ color: var(--muted); font-size: 13px; margin-bottom: 28px; }}

  /* ── Building Tabs ── */
  .tabs {{ display: flex; gap: 6px; margin-bottom: 24px; flex-wrap: wrap; }}
  .tab-btn {{
    padding: 8px 18px; border-radius: 8px; border: 1px solid var(--border);
    background: var(--surface); color: var(--muted); cursor: pointer;
    font-size: 13px; font-weight: 600; transition: all .2s; font-family: inherit;
  }}
  .tab-btn.active, .tab-btn:hover {{
    background: var(--accent); color: #fff; border-color: var(--accent);
  }}
  .tab-panel {{ display: none; }}
  .tab-panel.active {{ display: block; }}

  /* ── KPI Cards ── */
  .kpi-grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(160px, 1fr)); gap: 14px; margin-bottom: 28px; }}
  .kpi {{
    background: var(--surface); border: 1px solid var(--border);
    border-radius: 12px; padding: 18px 20px;
    transition: border-color .2s;
  }}
  .kpi:hover {{ border-color: var(--accent); }}
  .kpi-label {{ font-size: 11px; text-transform: uppercase; letter-spacing: 1px; color: var(--muted); margin-bottom: 6px; }}
  .kpi-value {{ font-size: 26px; font-weight: 700; font-family: var(--font-head); }}
  .kpi-value.green  {{ color: var(--accent);  }}
  .kpi-value.blue   {{ color: var(--accent2); }}
  .kpi-value.orange {{ color: var(--accent3); }}
  .kpi-value.red    {{ color: var(--danger);  }}
  .kpi-sub {{ font-size: 11px; color: var(--muted); margin-top: 4px; }}

  /* ── Chart containers ── */
  .chart-grid-2 {{ display: grid; grid-template-columns: 1fr 1fr; gap: 20px; margin-bottom: 20px; }}
  .chart-grid-3 {{ display: grid; grid-template-columns: repeat(3, 1fr); gap: 20px; margin-bottom: 20px; }}
  @media (max-width: 768px) {{
    .chart-grid-2, .chart-grid-3 {{ grid-template-columns: 1fr; }}
  }}
  .card {{
    background: var(--surface); border: 1px solid var(--border);
    border-radius: 14px; padding: 20px;
  }}
  .card-title {{ font-size: 13px; font-weight: 600; color: var(--muted); margin-bottom: 14px; text-transform: uppercase; letter-spacing: .5px; }}
  .chart-wrap {{ position: relative; height: 220px; }}
  .chart-wrap-tall {{ position: relative; height: 280px; }}
  .chart-wrap-full {{ position: relative; height: 200px; }}

  /* ── Coefficient table ── */
  .coef-table {{ width: 100%; border-collapse: collapse; font-size: 13px; }}
  .coef-table th {{
    text-align: left; padding: 8px 12px;
    border-bottom: 1px solid var(--border);
    color: var(--muted); font-weight: 600; font-size: 11px; text-transform: uppercase; letter-spacing: .5px;
  }}
  .coef-table td {{ padding: 8px 12px; border-bottom: 1px solid rgba(46,51,80,.5); }}
  .coef-table tr:last-child td {{ border-bottom: none; }}
  .sig {{ font-weight: 700; font-size: 12px; }}
  .sig-high {{ color: var(--accent); }}
  .sig-mid  {{ color: var(--accent2); }}
  .sig-low  {{ color: var(--muted); }}
  /* ── Coefficient bar（雙向，以中線為基準）── */
  .coef-cell {{ white-space: nowrap; vertical-align: middle; }}
  .coef-bar-wrap {{
    display: inline-flex; align-items: center; gap: 0;
    width: 200px; vertical-align: middle; margin-left: 12px;
  }}
  /* 左半：負值區，內容靠右對齊 */
  .coef-bar-left {{
    width: 100px; display: flex; justify-content: flex-end; align-items: center;
  }}
  /* 中線 */
  .coef-bar-center {{
    width: 2px; height: 12px; background: #4a5568; flex-shrink: 0;
  }}
  /* 右半：正值區，內容靠左對齊 */
  .coef-bar-right {{
    width: 100px; display: flex; justify-content: flex-start; align-items: center;
  }}
  .coef-bar-fill {{
    height: 8px; border-radius: 2px; opacity: .8;
  }}

  /* ── Scenario ── */
  .scenario-hero {{
    background: linear-gradient(135deg, rgba(63,17,99,.1), rgba(230,146,43,.06));
    border: 1px solid rgba(230,146,43,.25);
    border-radius: 16px; padding: 28px 32px; margin-bottom: 28px;
    text-align: center;
  }}
  .scenario-big {{ font-size: 48px; font-weight: 900; font-family: var(--font-head); color: var(--accent); }}
  .scenario-label {{ color: var(--muted); font-size: 14px; margin-top: 4px; }}
  .scenario-slider {{ margin: 24px 0 12px; }}
  .scenario-slider label {{ font-size: 13px; color: var(--muted); }}
  .scenario-slider input {{ width: 100%; cursor: pointer; accent-color: var(--accent); margin-top: 8px; }}
  .scenario-cards {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr)); gap: 14px; }}
  .sc-card {{
    background: var(--surface); border: 1px solid var(--border);
    border-radius: 12px; padding: 18px; text-align: center; transition: all .2s;
  }}
  .sc-card:hover {{ border-color: var(--accent); transform: translateY(-2px); }}
  .sc-n {{ font-size: 13px; color: var(--muted); margin-bottom: 6px; }}
  .sc-kwh {{ font-size: 22px; font-weight: 700; color: var(--accent); }}
  .sc-ntd {{ font-size: 13px; color: var(--accent3); margin-top: 4px; }}
  .sc-day {{ font-size: 11px; color: var(--muted); margin-top: 4px; }}

  /* ── Overview comparison ── */
  .overview-grid {{ display: grid; grid-template-columns: repeat(3, 1fr); gap: 16px; }}
  @media (max-width: 600px) {{ .overview-grid {{ grid-template-columns: 1fr; }} }}
  .ov-card {{
    background: var(--surface); border: 1px solid var(--border);
    border-radius: 14px; padding: 20px 22px;
  }}
  .ov-card-title {{ font-size: 16px; font-weight: 700; color: #fff; margin-bottom: 14px; }}
  .ov-row {{ display: flex; justify-content: space-between; align-items: center; padding: 6px 0; border-bottom: 1px solid rgba(46,51,80,.4); }}
  .ov-row:last-child {{ border-bottom: none; }}
  .ov-key {{ font-size: 12px; color: var(--muted); }}
  .ov-val {{ font-size: 13px; font-weight: 600; }}

  /* ── Footer ── */
  footer {{
    padding: 32px 40px; color: var(--muted); font-size: 12px;
    border-top: 1px solid var(--border); text-align: center;
  }}

  /* ── Animations ── */
  @keyframes fadeUp {{
    from {{ opacity: 0; transform: translateY(16px); }}
    to   {{ opacity: 1; transform: translateY(0); }}
  }}
  .fade-in {{ animation: fadeUp .5s ease both; }}
  .fade-in-1 {{ animation-delay: .05s; }}
  .fade-in-2 {{ animation-delay: .1s; }}
  .fade-in-3 {{ animation-delay: .15s; }}
</style>
</head>
<body>

<!-- Hero -->
<div class="hero fade-in">
  <div class="hero-label">NTU Energy Analytics</div>
  <h1>校舍用電分析報告</h1>
  <p>基於 OLS 固定效應模型的建築用電預測與情境模擬分析，涵蓋普通、新生、共同教學館。</p>
  <div class="hero-badges">
    <span class="badge badge-green">交乘項模型</span>
    <span class="badge badge-blue">月份 × 時段固定效應</span>
    <span class="badge badge-org">Out-of-Sample 驗證</span>
  </div>
</div>

<!-- Nav -->
<nav id="main-nav">
  <a href="#overview" class="active">總覽</a>
  <a href="#performance">模型效能</a>
  <a href="#prediction">預測分析</a>
  <a href="#residual">殘差診斷</a>
  <a href="#coefficients">係數解讀</a>
  <a href="#scenario">情境模擬</a>
</nav>

<div class="container">

<!-- ══════════════════════════════════════════════════════ -->
<!-- 0. OVERVIEW -->
<!-- ══════════════════════════════════════════════════════ -->
<section id="overview">
  <div class="section-title">三棟建築總覽</div>
  <div class="section-sub">Out-of-Sample 表現摘要，點擊各棟查看詳細分析</div>
  <div class="overview-grid" id="overview-grid"></div>
</section>

<!-- ══════════════════════════════════════════════════════ -->
<!-- 1. MODEL PERFORMANCE -->
<!-- ══════════════════════════════════════════════════════ -->
<section id="performance">
  <div class="section-title">模型效能</div>
  <div class="section-sub">比較各棟建築的 In-sample（訓練）與 Out-of-sample（測試）擬合品質</div>

  <div class="tabs" id="perf-tabs"></div>
  <div id="perf-panels"></div>
</section>

<!-- ══════════════════════════════════════════════════════ -->
<!-- 2. PREDICTION TIME SERIES -->
<!-- ══════════════════════════════════════════════════════ -->
<section id="prediction">
  <div class="section-title">實際 vs 預測用電</div>
  <div class="section-sub">測試期日均用電，虛線為模型預測值</div>
  <div class="tabs" id="pred-tabs"></div>
  <div id="pred-panels"></div>
</section>

<!-- ══════════════════════════════════════════════════════ -->
<!-- 3. RESIDUAL DIAGNOSTICS -->
<!-- ══════════════════════════════════════════════════════ -->
<section id="residual">
  <div class="section-title">殘差診斷</div>
  <div class="section-sub">判斷模型是否存在系統性偏差</div>
  <div class="tabs" id="resid-tabs"></div>
  <div id="resid-panels"></div>
</section>

<!-- ══════════════════════════════════════════════════════ -->
<!-- 4. COEFFICIENTS -->
<!-- ══════════════════════════════════════════════════════ -->
<section id="coefficients">
  <div class="section-title">核心係數解讀</div>
  <div class="section-sub">每增加一個單位對應的用電變化量（kWh），僅顯示核心變數</div>
  <div class="tabs" id="coef-tabs"></div>
  <div id="coef-panels"></div>
</section>

<!-- ══════════════════════════════════════════════════════ -->
<!-- 5. SCENARIO -->
<!-- ══════════════════════════════════════════════════════ -->
<section id="scenario">
  <div class="section-title">情境模擬：少開大教室節電效益</div>
  <div class="section-sub">基於模型係數推算，關閉 N 間大教室在測試學期的預期節電效益</div>
  <div class="tabs" id="scen-tabs"></div>
  <div id="scen-panels"></div>
</section>

</div><!-- /container -->

<footer>
  本報告由 <strong>generate_report.py</strong> 自動生成 ·
  模型：OLS + 交乘項 + 月份/時段固定效應
</footer>

<!-- ════════════ JavaScript ════════════ -->
<script>
const DATA = {data_json};
const BUILDINGS = Object.keys(DATA);
const CH = {{}};
BUILDINGS.forEach(k => CH[k] = DATA[k].ch);

const C = {{
  actual  : '#9d94c0',
  pred    : '#e6922b',
  resid   : '#fedfb2',
  scatter : '#9d94c0',
  grid    : 'rgba(63,17,99,.4)',
  tick    : '#9d94c0',
}};

Chart.defaults.color = C.tick;
Chart.defaults.borderColor = C.grid;
Chart.defaults.font.family = "'Helvetica Neue','PingFang TC',sans-serif";

const charts = {{}};
function destroyChart(id) {{ if (charts[id]) {{ charts[id].destroy(); delete charts[id]; }} }}

/* ── Helpers ── */
function makeTabs(containerId, panelId, keys, labelFn, contentFn, activeCls='active') {{
  const tabEl = document.getElementById(containerId);
  const panEl = document.getElementById(panelId);
  tabEl.innerHTML = ''; panEl.innerHTML = '';

  keys.forEach((k, i) => {{
    const btn = document.createElement('button');
    btn.className = 'tab-btn' + (i===0 ? ' active' : '');
    btn.textContent = labelFn(k);
    btn.onclick = () => {{
      tabEl.querySelectorAll('.tab-btn').forEach(b => b.classList.remove('active'));
      panEl.querySelectorAll('.tab-panel').forEach(p => p.classList.remove('active'));
      btn.classList.add('active');
      document.getElementById(panelId + '_' + k).classList.add('active');
    }};
    tabEl.appendChild(btn);

    const panel = document.createElement('div');
    panel.className = 'tab-panel' + (i===0 ? ' active' : '');
    panel.id = panelId + '_' + k;
    panel.innerHTML = contentFn(k);
    panEl.appendChild(panel);
  }});
}}

function r2Color(v) {{
  if (v >= 0.9) return '#e6922b';
  if (v >= 0.75) return '#9d94c0';
  return '#fedfb2';
}}

/* ══════════ 0. OVERVIEW ══════════ */
(function buildOverview() {{
  const grid = document.getElementById('overview-grid');
  BUILDINGS.forEach(en => {{
    const d = DATA[en]; const m = d.metrics;
    const card = document.createElement('div');
    card.className = 'ov-card fade-in';
    const rows = [
      ['In-sample R²',    m.r2_in.toFixed(3),   r2Color(m.r2_in)],
      ['OOS R²',          m.r2_oos.toFixed(3),   r2Color(m.r2_oos)],
      ['RMSE (訓練)',      m.rmse_in.toFixed(2) + ' kWh', '#9d94c0'],
      ['RMSE (測試)',      m.rmse_oos.toFixed(2) + ' kWh', '#9d94c0'],
      ['訓練筆數',         m.n_train.toLocaleString(), '#9d94c0'],
      ['測試筆數',         m.n_test.toLocaleString(),  '#9d94c0'],
    ];
    const partialBanner = d.is_partial
      ? `<div style="background:rgba(230,146,43,.12);border:1px solid rgba(230,146,43,.35);
                     border-radius:8px;padding:8px 12px;margin-bottom:12px;
                     font-size:11px;color:#e6922b;line-height:1.5">
           ⚠ 測試資料僅含 11–12 月<br>（9–10 月因資料品質已排除）
         </div>`
      : '';
    card.innerHTML = `<div class="ov-card-title">${{d.ch}} 教學館</div>` +
      partialBanner +
      rows.map(([k,v,c]) =>
        `<div class="ov-row"><span class="ov-key">${{k}}</span>
         <span class="ov-val" style="color:${{c}}">${{v}}</span></div>`
      ).join('');
    grid.appendChild(card);
  }});
}})();

/* ══════════ 1. PERFORMANCE ══════════ */
makeTabs('perf-tabs', 'perf-panels', BUILDINGS, en => CH[en] + ' 教學館', en => {{
  const m = DATA[en].metrics;
  return `
  <div class="kpi-grid">
    <div class="kpi"><div class="kpi-label">In-sample R²</div>
      <div class="kpi-value" style="color:${{r2Color(m.r2_in)}}">${{m.r2_in.toFixed(3)}}</div>
      <div class="kpi-sub">訓練資料擬合度</div></div>
    <div class="kpi"><div class="kpi-label">Out-of-sample R²</div>
      <div class="kpi-value" style="color:${{r2Color(m.r2_oos)}}">${{m.r2_oos.toFixed(3)}}</div>
      <div class="kpi-sub">未見資料預測力</div></div>
    <div class="kpi"><div class="kpi-label">RMSE 訓練</div>
      <div class="kpi-value blue">${{m.rmse_in.toFixed(2)}}</div>
      <div class="kpi-sub">kWh</div></div>
    <div class="kpi"><div class="kpi-label">RMSE 測試</div>
      <div class="kpi-value orange">${{m.rmse_oos.toFixed(2)}}</div>
      <div class="kpi-sub">kWh</div></div>
    <div class="kpi"><div class="kpi-label">訓練樣本數</div>
      <div class="kpi-value" style="color:#94a3b8">${{m.n_train.toLocaleString()}}</div></div>
    <div class="kpi"><div class="kpi-label">測試樣本數</div>
      <div class="kpi-value" style="color:#94a3b8">${{m.n_test.toLocaleString()}}</div></div>
  </div>
  <div class="card">
    <div class="card-title">R² 比較（訓練 vs 測試）</div>
    <div class="chart-wrap"><canvas id="perf_bar_${{en}}"></canvas></div>
  </div>`;
}});

BUILDINGS.forEach(en => {{
  const m = DATA[en].metrics;
  const ctx = document.getElementById('perf_bar_' + en);
  if (!ctx) return;
  destroyChart('perf_bar_' + en);
  charts['perf_bar_' + en] = new Chart(ctx, {{
    type: 'bar',
    data: {{
      labels: ['In-sample R²', 'OOS R²', 'RMSE 訓練 (÷10)', 'RMSE 測試 (÷10)'],
      datasets: [{{
        label: '數值',
        data: [m.r2_in, m.r2_oos, m.rmse_in/10, m.rmse_oos/10],
        backgroundColor: ['#3f116388','#e6922b88','#9d94c088','#fedfb288'],
        borderColor:     ['#3f1163',  '#e6922b',  '#9d94c0',  '#fedfb2'],
        borderWidth: 1.5, borderRadius: 6,
      }}]
    }},
    options: {{
      indexAxis: 'y', responsive: true, maintainAspectRatio: false,
      plugins: {{ legend: {{ display: false }}, tooltip: {{
        callbacks: {{ label: ctx => ' ' + ctx.parsed.x.toFixed(3) }}
      }} }},
      scales: {{ x: {{ max: 1.05, grid: {{ color: C.grid }} }}, y: {{ grid: {{ display: false }} }} }}
    }}
  }});
}});

/* ══════════ 2. PREDICTION ══════════ */
makeTabs('pred-tabs', 'pred-panels', BUILDINGS, en => CH[en] + ' 教學館', en => `
  <div class="chart-grid-2">
    <div class="card" style="grid-column:1/-1">
      <div class="card-title">日均用電：實際 vs 預測（測試學期）</div>
      <div class="chart-wrap-tall"><canvas id="pred_ts_${{en}}"></canvas></div>
    </div>
    <div class="card">
      <div class="card-title">散佈圖：實際 vs 預測</div>
      <div class="chart-wrap"><canvas id="pred_sc_${{en}}"></canvas></div>
    </div>
    <div class="card">
      <div class="card-title">殘差時序（日均）</div>
      <div class="chart-wrap"><canvas id="pred_re_${{en}}"></canvas></div>
    </div>
  </div>`
);

BUILDINGS.forEach(en => {{
  const d = DATA[en].daily;
  if (!d.Date || d.Date.length === 0) return;

  // Time series
  const tsCtx = document.getElementById('pred_ts_' + en);
  if (tsCtx) {{
    destroyChart('pred_ts_' + en);
    charts['pred_ts_' + en] = new Chart(tsCtx, {{
      type: 'line',
      data: {{
        labels: d.Date,
        datasets: [
          {{ label: '實際', data: d.Actual, borderColor: C.actual, backgroundColor: 'transparent',
             borderWidth: 1.5, pointRadius: 0, tension: .3 }},
          {{ label: '預測', data: d.Predicted, borderColor: C.pred, backgroundColor: 'rgba(74,222,128,.08)',
             borderWidth: 1.5, pointRadius: 0, tension: .3, borderDash: [5,3], fill: false }},
        ]
      }},
      options: {{
        responsive: true, maintainAspectRatio: false,
        plugins: {{ legend: {{ position: 'top', labels: {{ boxWidth: 12 }} }} }},
        scales: {{
          x: {{ ticks: {{ maxTicksLimit: 8, maxRotation: 30 }}, grid: {{ color: C.grid }} }},
          y: {{ title: {{ display: true, text: 'kWh' }}, grid: {{ color: C.grid }} }}
        }}
      }}
    }});
  }}

  // Scatter (sample 500 for perf)
  const sc = DATA[en].scatter;
  const step = Math.max(1, Math.floor(sc.actual.length / 500));
  const sAct = sc.actual.filter((_,i)=>i%step===0);
  const sPre = sc.predicted.filter((_,i)=>i%step===0);
  const minV = Math.min(...sAct, ...sPre);
  const maxV = Math.max(...sAct, ...sPre);
  const scCtx = document.getElementById('pred_sc_' + en);
  if (scCtx) {{
    destroyChart('pred_sc_' + en);
    charts['pred_sc_' + en] = new Chart(scCtx, {{
      type: 'scatter',
      data: {{
        datasets: [
          {{ label: '資料點', data: sAct.map((a,i)=>({{x:a,y:sPre[i]}})),
             backgroundColor: C.scatter + '44', pointRadius: 3 }},
          {{ label: '45°線', data: [{{x:minV,y:minV}},{{x:maxV,y:maxV}}],
             type:'line', borderColor:'#fedfb2', borderWidth:1.5, pointRadius:0, borderDash:[4,3] }}
        ]
      }},
      options: {{
        responsive: true, maintainAspectRatio: false,
        plugins: {{ legend: {{ position:'top', labels:{{ boxWidth:10 }} }} }},
        scales: {{
          x: {{ title:{{ display:true, text:'實際 (kWh)' }}, grid:{{ color:C.grid }} }},
          y: {{ title:{{ display:true, text:'預測 (kWh)' }}, grid:{{ color:C.grid }} }}
        }}
      }}
    }});
  }}

  // Residual time series
  const reCtx = document.getElementById('pred_re_' + en);
  if (reCtx) {{
    destroyChart('pred_re_' + en);
    charts['pred_re_' + en] = new Chart(reCtx, {{
      type: 'line',
      data: {{
        labels: d.Date,
        datasets: [{{
          label: '殘差', data: d.Residual, borderColor: C.resid,
          backgroundColor: 'rgba(251,146,60,.08)', fill: true,
          borderWidth: 1.2, pointRadius: 0, tension: .3
        }}]
      }},
      options: {{
        responsive: true, maintainAspectRatio: false,
        plugins: {{ legend: {{ display: false }} }},
        scales: {{
          x: {{ ticks:{{ maxTicksLimit:6 }}, grid:{{ color:C.grid }} }},
          y: {{ title:{{ display:true, text:'殘差 (kWh)' }},
               grid:{{ color:C.grid }},
               ticks:{{ callback: v => (v>0?'+':'') + v.toFixed(1) }} }}
        }}
      }}
    }});
  }}
}});

/* ══════════ 3. RESIDUAL DIAGNOSTICS ══════════ */
makeTabs('resid-tabs', 'resid-panels', BUILDINGS, en => CH[en] + ' 教學館', en => `
  <div class="chart-grid-2">
    <div class="card">
      <div class="card-title">殘差分佈直方圖</div>
      <div class="chart-wrap"><canvas id="resid_hist_${{en}}"></canvas></div>
    </div>
    <div class="card">
      <div class="card-title">殘差 vs 預測值（同質性檢驗）</div>
      <div class="chart-wrap"><canvas id="resid_fit_${{en}}"></canvas></div>
    </div>
  </div>`
);

BUILDINGS.forEach(en => {{
  const h = DATA[en].hist;

  // Histogram
  const hCtx = document.getElementById('resid_hist_' + en);
  if (hCtx && h.counts.length > 0) {{
    const labels = h.edges.slice(0,-1).map((v,i) => ((v + h.edges[i+1])/2).toFixed(2));
    destroyChart('resid_hist_' + en);
    charts['resid_hist_' + en] = new Chart(hCtx, {{
      type: 'bar',
      data: {{
        labels,
        datasets: [{{ label: '頻次', data: h.counts,
          backgroundColor: '#9d94c055', borderColor: '#9d94c0', borderWidth: 1 }}]
      }},
      options: {{
        responsive: true, maintainAspectRatio: false,
        plugins: {{ legend: {{ display:false }} }},
        scales: {{
          x: {{ ticks:{{ maxTicksLimit:8, maxRotation:0 }}, grid:{{ color:C.grid }} }},
          y: {{ title:{{ display:true, text:'次數' }}, grid:{{ color:C.grid }} }}
        }}
      }}
    }});
  }}

  // Residual vs Fitted (scatter, sampled)
  const sc = DATA[en].scatter;
  const step = Math.max(1, Math.floor(sc.predicted.length / 400));
  const pred_s = sc.predicted.filter((_,i)=>i%step===0);
  // residuals from scatter
  const resid_s = DATA[en].scatter.actual
    .filter((_,i)=>i%step===0)
    .map((a,i) => a - pred_s[i]);

  const fCtx = document.getElementById('resid_fit_' + en);
  if (fCtx) {{
    destroyChart('resid_fit_' + en);
    charts['resid_fit_' + en] = new Chart(fCtx, {{
      type: 'scatter',
      data: {{
        datasets: [
          {{ label: '殘差', data: pred_s.map((p,i)=>({{x:p,y:resid_s[i]}})),
             backgroundColor: '#fedfb244', pointRadius: 3 }},
          {{ label: '零線', data: [
              {{x:Math.min(...pred_s), y:0}}, {{x:Math.max(...pred_s), y:0}}],
             type:'line', borderColor:'#fedfb2', borderWidth:1.5, pointRadius:0, borderDash:[4,3] }}
        ]
      }},
      options: {{
        responsive: true, maintainAspectRatio: false,
        plugins: {{ legend: {{ position:'top', labels:{{ boxWidth:10 }} }} }},
        scales: {{
          x: {{ title:{{ display:true, text:'預測值 (kWh)' }}, grid:{{ color:C.grid }} }},
          y: {{ title:{{ display:true, text:'殘差 (kWh)' }}, grid:{{ color:C.grid }} }}
        }}
      }}
    }});
  }}
}});

/* ══════════ 4. COEFFICIENTS ══════════ */
makeTabs('coef-tabs', 'coef-panels', BUILDINGS, en => CH[en] + ' 教學館', en => {{
  const coefs = DATA[en].coefs;
  if (!coefs || coefs.length === 0) return '<p style="color:var(--muted);padding:20px">無係數資料</p>';

  // 排除 const，以非常數項的最大絕對值為基準，const 不參與 bar 縮放
  const nonConst = coefs.filter(c => c.Variable !== 'const');
  const maxCoef  = Math.max(...nonConst.map(c => Math.abs(c.Coef)), 0.0001);
  const HALF_PX  = 96; // 中線到最遠端的最大 px

  const rows = coefs.map(c => {{
    const isConst  = c.Variable === 'const';
    const dir      = c.Coef >= 0 ? '#e6922b' : '#9d94c0';
    const sigClass = c.sig === '***' ? 'sig-high' : (c.sig === 'n.s.' ? 'sig-low' : 'sig-mid');
    const coefStr  = (c.Coef >= 0 ? '+' : '') + c.Coef.toFixed(4);
    const varStyle = isConst
      ? 'font-weight:600;color:#94a3b8;font-style:italic'
      : 'font-weight:600;color:#e2e8f0';

    // bar 長度（px）
    const barPx = isConst ? 0 : Math.min(Math.abs(c.Coef) / maxCoef * HALF_PX, HALF_PX);

    // 雙向 bar：左半放負值，右半放正值，中線固定
    const barHtml = `
      <span class="coef-bar-wrap">
        <span class="coef-bar-left">
          ${{c.Coef < 0 && !isConst
            ? `<span class="coef-bar-fill" style="width:${{barPx}}px;background:${{dir}}"></span>`
            : ''}}
        </span>
        <span class="coef-bar-center"></span>
        <span class="coef-bar-right">
          ${{c.Coef >= 0 && !isConst
            ? `<span class="coef-bar-fill" style="width:${{barPx}}px;background:${{dir}}"></span>`
            : ''}}
        </span>
      </span>`;

    return `<tr>
      <td style="${{varStyle}}">${{c.Variable}}</td>
      <td class="coef-cell">
        <span style="color:${{dir}};font-weight:700;min-width:72px;display:inline-block">${{coefStr}}</span>
        ${{barHtml}}
      </td>
      <td style="color:var(--muted)">${{c.StdErr.toFixed(4)}}</td>
      <td style="color:var(--muted)">${{c.P_value < 0.001 ? '<0.001' : c.P_value.toFixed(4)}}</td>
      <td class="sig ${{sigClass}}">${{c.sig}}</td>
    </tr>`;
  }}).join('');

  return `
  <!-- ── Model Specification ── -->
  <div class="card" style="margin-bottom:20px">
    <div class="card-title">Model Specification</div>

    <!-- 公式區 -->
    <div style="background:var(--surface2);border:1px solid var(--border);border-radius:10px;
                padding:14px 18px;margin-bottom:16px;font-family:'Georgia',serif;
                font-size:13px;line-height:2;color:var(--text);overflow-x:auto;white-space:nowrap">
      <span style="color:var(--accent);font-weight:700">Electricity<sub>it</sub></span>
      &nbsp;=&nbsp;
      <span style="color:var(--muted)">β<sub>0</sub></span>
      &nbsp;+&nbsp;
      <span style="color:#9d94c0">β<sub>1</sub></span>·Temp_c<sub>it</sub>
      &nbsp;+&nbsp;
      <span style="color:#e6922b">β<sub>2</sub></span>·BigC<sub>it</sub>
      &nbsp;+&nbsp;
      <span style="color:#e6922b">β<sub>3</sub></span>·SmallC<sub>it</sub>
      &nbsp;+&nbsp;
      <span style="color:#fedfb2">β<sub>4</sub></span>·(Temp_c × BigC)<sub>it</sub>
      &nbsp;+&nbsp;
      <span style="color:#fedfb2">β<sub>5</sub></span>·(Temp_c × SmallC)<sub>it</sub>
      &nbsp;+&nbsp;
      <span style="color:#c4b0d8">γ<sub>m</sub></span>
      &nbsp;+&nbsp;
      <span style="color:#c4b0d8">δ<sub>h</sub></span>
      &nbsp;+&nbsp;
      ε<sub>it</sub>
    </div>

    <!-- 變數說明 badges -->
    <div style="display:flex;flex-wrap:wrap;gap:10px;margin-bottom:16px">
      <div style="background:rgba(157,148,192,.1);border:1px solid rgba(157,148,192,.3);
                  border-radius:8px;padding:8px 12px;min-width:180px">
        <div style="font-size:10px;color:#9d94c0;text-transform:uppercase;letter-spacing:1px;margin-bottom:4px">
          溫度項
        </div>
        <div style="font-size:12px;color:var(--text);line-height:1.6">
          <b>Temp_c</b>：氣溫 − 25°C（中心化）<br>
        </div>
      </div>
      <div style="background:rgba(230,146,43,.1);border:1px solid rgba(230,146,43,.3);
                  border-radius:8px;padding:8px 12px;min-width:180px">
        <div style="font-size:10px;color:#e6922b;text-transform:uppercase;letter-spacing:1px;margin-bottom:4px">
          教室使用
        </div>
        <div style="font-size:12px;color:var(--text);line-height:1.6">
          <b>BigC</b>：同時段大教室使用間數<br>
          <b>SmallC</b>：同時段小教室使用間數
        </div>
      </div>
      <div style="background:rgba(254,223,178,.1);border:1px solid rgba(254,223,178,.3);
                  border-radius:8px;padding:8px 12px;min-width:180px">
        <div style="font-size:10px;color:#fedfb2;text-transform:uppercase;letter-spacing:1px;margin-bottom:4px">
          交乘項
        </div>
        <div style="font-size:12px;color:var(--text);line-height:1.6">
          <b>Temp_c × BigC</b>：溫度對大教室用電的異質效果<br>
          <b>Temp_c × SmallC</b>：溫度對小教室用電的異質效果
        </div>
      </div>
      <div style="background:rgba(63,17,99,.2);border:1px solid rgba(63,17,99,.5);
                  border-radius:8px;padding:8px 12px;min-width:180px">
        <div style="font-size:10px;color:#c4b0d8;text-transform:uppercase;letter-spacing:1px;margin-bottom:4px">
          Fixed Effects
        </div>
        <div style="font-size:12px;color:var(--text);line-height:1.6">
          <b>γ<sub>m</sub></b>：月份固定效應（Month FE）<br>
          <b>δ<sub>h</sub></b>：時段固定效應（Hour FE）
        </div>
      </div>
    </div>

    <!-- 補充說明 -->
    <div style="font-size:11px;color:var(--muted);line-height:1.8;border-top:1px solid var(--border);padding-top:10px">
      訓練期：2024/09–12（上學期）&nbsp;·&nbsp;
      估計方法：OLS&nbsp;·&nbsp;
      排除國定假日與補假&nbsp;·&nbsp;
      僅納入第 1–10 節上課時段
    </div>
  </div>

  <!-- ── 係數表 ── -->
  <div class="card">
    <div style="margin-bottom:12px;font-size:12px;color:var(--muted)">
      *** p&lt;0.001 &nbsp;** p&lt;0.01 &nbsp;* p&lt;0.05 &nbsp;n.s. 不顯著 &nbsp;·&nbsp;
      <span style="color:#e6922b">▌</span> 正向用電 &nbsp;
      <span style="color:#9d94c0">▌</span> 負向用電
    </div>
    <table class="coef-table">
      <thead><tr>
        <th>變數</th>
        <th>
          係數
          <span style="display:inline-flex;width:200px;vertical-align:middle;margin-left:12px;font-weight:400;font-size:10px;color:var(--muted)">
            <span style="width:100px;text-align:right;padding-right:6px">← 負</span>
            <span style="width:2px"></span>
            <span style="width:100px;text-align:left;padding-left:6px">正 →</span>
          </span>
        </th>
        <th>Std Err</th><th>P-value</th><th>顯著</th>
      </tr></thead>
      <tbody>${{rows}}</tbody>
    </table>
  </div>`;
}});

/* ══════════ 5. SCENARIO ══════════ */
makeTabs('scen-tabs', 'scen-panels', BUILDINGS, en => CH[en] + ' 教學館', en => {{
  const sc = DATA[en].scenario;
  if (!sc || sc.length === 0) return '<p style="color:var(--muted);padding:20px">無情境資料</p>';

  const isPartial  = DATA[en].is_partial;
  const maxReduce  = DATA[en].bigc_max;
  const scFiltered = sc.filter(s => s.reduce_n <= maxReduce);
  const heroItem   = scFiltered[scFiltered.length - 1];
  const testDays   = scFiltered[0].test_days;

  // 建築名稱下方的期間說明（三棟都有，普通館橘色警示）
  const periodNote = isPartial
    ? `<div style="font-size:12px;color:#e6922b;font-style:italic;margin-bottom:20px;
                   display:flex;align-items:center;gap:6px">
         <span>⚠</span>
         <span>資料僅含 11–12 月（共 ${{testDays}} 天）— 9–10 月因品質問題已排除，<strong>不宜直接與其他棟比較</strong></span>
       </div>`
    : `<div style="font-size:12px;color:var(--muted);margin-bottom:20px">
         完整學期，共 ${{testDays}} 天
       </div>`;

  const filteredCards = scFiltered.map(s => `
    <div class="sc-card">
      <div class="sc-n">少開 <strong style="color:#fff;font-size:18px">${{s.reduce_n}}</strong> 間大教室</div>
      <div class="sc-kwh">${{s.total_kwh.toLocaleString(undefined,{{maximumFractionDigits:0}})}} kWh</div>
      <div class="sc-day">測試期 ${{s.test_days}} 天 · 日均省 ${{s.avg_per_day.toFixed(1)}} kWh</div>
    </div>`).join('');

  return `
  ${{periodNote}}
  <div class="scenario-hero">
    <div style="font-size:12px;color:var(--muted);letter-spacing:2px;text-transform:uppercase;margin-bottom:8px">
      測試期最大節電潛力（少開 ${{heroItem.reduce_n}} 間）
    </div>
    <div class="scenario-big">${{heroItem.total_kwh.toLocaleString(undefined,{{maximumFractionDigits:0}})}} kWh</div>
    <div class="scenario-label">相當於減少 ${{(heroItem.total_kwh * 0.509).toFixed(0)}} kg CO₂</div>
  </div>
  <div class="scenario-cards">${{filteredCards}}</div>
  <div class="card" style="margin-top:20px">
    <div class="card-title">節電效益比較</div>
    <div class="chart-wrap"><canvas id="scen_bar_${{en}}"></canvas></div>
  </div>`;
}});

BUILDINGS.forEach(en => {{
  const sc = DATA[en].scenario;
  if (!sc || sc.length === 0) return;
  const bCtx = document.getElementById('scen_bar_' + en);
  if (!bCtx) return;
  const maxR = DATA[en].bigc_max;
  const scF  = sc.filter(s => s.reduce_n <= maxR);
  const ns   = scF.map(s => `少開 ${{s.reduce_n}} 間`);
  const kwhs = scF.map(s => s.total_kwh);
  destroyChart('scen_bar_' + en);
  charts['scen_bar_' + en] = new Chart(bCtx, {{
    type: 'bar',
    data: {{
      labels: ns,
      datasets: [{{
        label: '省電量 (kWh)',
        data: kwhs,
        backgroundColor: ['#3f116366','#9d94c099','#fedfb266','#e6922bcc'],
        borderColor: '#e6922b', borderWidth: 1.5, borderRadius: 8,
      }}]
    }},
    options: {{
      responsive: true, maintainAspectRatio: false,
      plugins: {{
        legend: {{ display: false }},
        tooltip: {{ callbacks: {{ label: ctx =>
          ` ${{ctx.parsed.y.toLocaleString()}} kWh` }} }}
      }},
      scales: {{
        x: {{ grid: {{ display: false }} }},
        y: {{ title: {{ display: true, text: 'kWh' }}, grid: {{ color: C.grid }} }}
      }}
    }}
  }});
}});

/* ══════════ Scrollspy ══════════ */
const sections = ['overview','performance','prediction','residual','coefficients','scenario'];
const navLinks  = document.querySelectorAll('nav a');
window.addEventListener('scroll', () => {{
  let cur = '';
  sections.forEach(id => {{
    const el = document.getElementById(id);
    if (el && window.scrollY >= el.offsetTop - 80) cur = id;
  }});
  navLinks.forEach(a => {{
    a.classList.toggle('active', a.getAttribute('href') === '#' + cur);
  }});
}}, {{ passive: true }});

</script>
</body>
</html>"""

# ── Write output ───────────────────────────────────────────────────────────────
out_path = os.path.join(PLOT_FOLDER, "electricity_report.html")
with open(out_path, 'w', encoding='utf-8') as f:
    f.write(HTML)

print(f"[完成] 互動報告已產出：{out_path}")
print(f"       直接用瀏覽器開啟即可，無需伺服器。")