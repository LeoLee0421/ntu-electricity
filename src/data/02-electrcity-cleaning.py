# Import packages ------------------------------------------------------------------------------
import pandas as pd
import os


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


# 處理廣設號碼對應資料 -------------------------------------------------------------------------------
building_num_dfs = pd.read_html(os.path.join(RAW_FOLDER, "館舍號碼對應.html"))

building_num = building_num_dfs[0]
building_num = building_num.iloc[:, [1, 3]]  
building_num = building_num.iloc[1:].reset_index(drop=True)
building_num.columns = ["建物編碼", "建物名稱"]

building_num.to_csv(os.path.join(PROCESSED_FOLDER, "館舍號碼對應.csv"), index=False)

target = ["新生教學館", "普通教學館", "綜合教學館", "博雅教學館", "共同教學館"]
target_building = building_num[building_num["建物名稱"].str.strip().isin(target)]
target_codes = target_building["建物編碼"]

code_to_english = {}
for _, row in target_building.iterrows():
    code = row["建物編碼"]
    full_name = row["建物名稱"].strip()
    
    # 尋找這個全名中包含哪個關鍵字 (例如 "普通教學館" 包含 "普通")
    for zh_key, en_val in BUILDING_MAP.items():
        if zh_key in full_name:
            code_to_english[code] = en_val
            break
        

# 處理電表資料 --------------------------------------------------------------------------------------
electricity_df_2024 = pd.read_csv(os.path.join(RAW_FOLDER, "2024_ele_all_buildings_V3.csv"))
electricity_df_2025 = pd.read_csv(os.path.join(RAW_FOLDER, "2025_ele_all_buildings_V3.csv"))
electricity_df = pd.concat([electricity_df_2024, electricity_df_2025], ignore_index=True)
electricity_df.set_index("DateTime", inplace=True)

target_electricity_df = electricity_df[target_codes]
target_electricity_df = target_electricity_df.rename(columns=code_to_english)

target_electricity_df.to_csv(os.path.join(PROCESSED_FOLDER, "target_electricity_dataframe.csv"))

print('Save to target_electricity_dataframe.csv successfully!')