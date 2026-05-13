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

# Classifying ------------------------------------------------------------------------------------

#def classify_capacity(cap):
#    if cap > 200:  return "BigC"
#    elif cap > 100: return "MediumC"
#    else:           return "SmallC"

def classify_capacity(cap):
    if cap >= 130:  return "BigC"
    else:           return "SmallC"    

all_building_inventory = []

for building, en_name in BUILDING_MAP.items():
    binary_path = os.path.join(PROCESSED_FOLDER, f"{en_name}-timetable-binary")
    csv_files = glob.glob(os.path.join(binary_path, "*.csv"))
    
    building_all_days = []
    # 建立一個變數來暫存最後一個 df，用於生成 room_list
    last_df_for_inventory = None 

    for file in tqdm(csv_files, desc=f"Aggregating {building}"):
        df = pd.read_csv(file)
        
        # 1. 剔除容納人數小於 10 人的教室
        df = df[df['capacity'] >= 10].copy()
        if df.empty: continue
        

        date_str = os.path.basename(file).replace(".csv", "").split("_")[-1]
        date_obj = datetime.strptime(date_str, "%Y-%m-%d")
        
        # 2. 分類
        df['cap_group'] = df['capacity'].apply(classify_capacity)
        last_df_for_inventory = df.copy() # 存下來供 inventory 使用
        
        if 'room' not in df.columns and df.index.name == 'room':
            df = df.reset_index()

        time_cols = [c for c in df.columns if c not in ['room', 'capacity', 'cap_group']]
        df_melted = df.melt(id_vars=['room', 'cap_group'], value_vars=time_cols, var_name='Time', value_name='inclass')
        df_melted['inclass'] = pd.to_numeric(df_melted['inclass'], errors='coerce').fillna(0).astype(int)
        
        # 3. 聚合
        df_agg = df_melted.groupby(['Time', 'cap_group'])['inclass'].sum().unstack(fill_value=0).reset_index()

        # 確保欄位存在（對齊你的 classify_capacity）
        for col in ["BigC", "SmallC"]:
            if col not in df_agg.columns:
                df_agg[col] = 0

        # 4. 加入日期資訊並存入列表 (這是你原本漏掉的關鍵步驟)
        df_agg['Date'] = date_str
        df_agg['DoW'] = date_obj.strftime("%a") # 星期幾
        building_all_days.append(df_agg)

    # Save data
    if building_all_days:
        final_df = pd.concat(building_all_days, ignore_index=True)
        
        # 根據你目前的 classify_capacity，移除 MediumC 以免噴錯
        final_df = final_df[['Date', 'DoW', 'Time', 'BigC', 'SmallC']]
        
        ntu_time_order = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9', '10', 'A', 'B', 'C', 'D']
        final_df['Time'] = pd.Categorical(final_df['Time'], categories=ntu_time_order, ordered=True)
        final_df = final_df.sort_values(['Date', 'Time'])

        # 確保儲存目錄存在
        output_dir = os.path.join(PROCESSED_FOLDER, "timetable-summary")
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        save_name = f"{en_name}_timetable_summary.csv"
        final_df.to_csv(os.path.join(output_dir, save_name), index=False)

        # 處理 inventory
        if last_df_for_inventory is not None:
            room_list = last_df_for_inventory[['room', 'capacity', 'cap_group']].drop_duplicates()
            room_list['building'] = building
            all_building_inventory.append(room_list)

if all_building_inventory:
    inventory_df = pd.concat(all_building_inventory, ignore_index=True)
    inventory_df = inventory_df[['building', 'room', 'capacity', 'cap_group']]
    inventory_df.columns = ['建築', '教室編號', '容納人數', '類型']
    inventory_df.to_csv(os.path.join(PROCESSED_FOLDER, "all_buildings_room_list.csv"), index=False, encoding='utf-8-sig')
    print("教室清單已生成：all_buildings_room_list.csv")

print("Done!")