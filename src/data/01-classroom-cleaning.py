# Import packages ------------------------------------------------------------------------------
import requests
import pandas as pd
from datetime import datetime, timedelta
import time
from tqdm import tqdm
import os
import glob
import traceback
import io


# Set paths ------------------------------------------------------------------------------------
WORK_DIR = os.getcwd()
RAW_FOLDER = os.path.join(WORK_DIR, "data", "raw")
PROCESSED_FOLDER = os.path.join(WORK_DIR, "data", "processed")

# 建築對應（中文 → 英文）
BUILDING_MAP = {
    "普通": "putong",
    "綜合": "zonghe",
    "博雅": "boya",
    "新生": "xinsheng",
    "共同": "gongtong"
}

BUILDING_PAGES = {
    "普通": 3,
    "綜合": 3,
    "博雅": 3,
    "共同": 3,
    "新生": 2
}

# 建立資料夾（raw & processed）
for en in BUILDING_MAP.values():
    os.makedirs(os.path.join(RAW_FOLDER, f"{en}-timetable"), exist_ok=True)
    os.makedirs(os.path.join(PROCESSED_FOLDER, f"{en}-timetable-binary"), exist_ok=True)


# Set request settings -------------------------------------------------------------------------
url = "https://gra206.aca.ntu.edu.tw/classrm/acarm/check-by-date1-new"
headers = {
    "User-Agent": "Mozilla/5.0",
    "Accept-Language": "zh-TW,zh-Hant;q=0.9"
}

# Set request days -----------------------------------------------------------------------------
start_date = datetime(2024, 9, 2)
end_date = datetime(2025, 12, 19)

dates = []
current_date = start_date
while current_date <= end_date:
    if current_date.weekday() < 5:
        dates.append(current_date)
    current_date += timedelta(days=1)


# Processing requests and cleaning data --------------------------------------------------------

for current_date in tqdm(dates, desc="Processing dates"):

    # Set time
    date_str = current_date.strftime("%Y-%m-%d")
    building_data = {b: [] for b in BUILDING_MAP.keys()}

    # Read in request files
    for building in BUILDING_MAP.keys():
        
        max_page = BUILDING_PAGES[building]

        for page in range(1, max_page + 1):

            params = {
                "page": page,
                "DateDDL": date_str,
                "BuildingDDL": building,
                "Capacity": 1,
                "SelectButton": "查詢"
            }

            response = requests.get(url, params=params, headers=headers)

            if response.status_code != 200:
                print(f"Failed on {date_str}, page {page}, building {building}")
                continue

            try:
                html_io = io.StringIO(response.text)
                dfs = pd.read_html(html_io, flavor="lxml", attrs={"id": "ClassTimeGV"})
                df = dfs[0]

                building_data[building].append(df)

            except Exception:
                print(f"Error on {date_str}, page {page}, building {building}:")
                traceback.print_exc()

    # Preprocessing + Save =========================================================
    for building, dfs_list in building_data.items():

        if not dfs_list:
            continue

        df = pd.concat(dfs_list, ignore_index=True)

        # Drop last row
        df = df.drop(df.index[-1])

        # Separate room and capacity（泛化版本）
        first_col = df.columns[0]
        df[['room', 'capacity']] = df[first_col].str.extract(r'(\D+\d+)\s*(\d+)人')
        df = df.dropna(subset=['room', 'capacity'])
        df['capacity'] = df['capacity'].astype(int)

        # Set room as index
        df = df.drop(columns=[first_col])
        df = df.set_index('room')

        # Set time period as columns
        cols = df.columns
        new_cols = []

        for col in cols:
            parts = str(col).split()

            if col == "capacity":
                continue

            if len(parts) == 3:
                start, end, period = parts
                new_cols.append(period)
            else:
                new_cols.append(col)

        df = df.drop(columns=["capacity"], errors="ignore")
        df.columns = new_cols

        # Save RAW ================================================================
        en_name = BUILDING_MAP[building]
        raw_path = os.path.join(RAW_FOLDER, f"{en_name}-timetable")

        filename = f"{en_name}_timetable_{date_str}.csv"
        df.to_csv(os.path.join(raw_path, filename))

    time.sleep(1)


# Save into binary tables ------------------------------------------------------------------------

for building, en_name in BUILDING_MAP.items():

    raw_path = os.path.join(RAW_FOLDER, f"{en_name}-timetable")
    processed_path = os.path.join(PROCESSED_FOLDER, f"{en_name}-timetable-binary")

    csv_files = glob.glob(os.path.join(raw_path, "*.csv"))

    for file in csv_files:

        df = pd.read_csv(file)
        df = df.set_index('room')
        df_binary = df.notna().astype(int)

        date_part = os.path.basename(file).split("_")[-1]

        filename = f"{en_name}_timetable_binary_{date_part}"
        df_binary.to_csv(os.path.join(processed_path, filename), index=True)


print("Done")