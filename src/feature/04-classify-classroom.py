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

def classify_capacity(cap):
    if cap > 200:  return "BigC"
    elif cap > 100: return "MediumC"
    else:           return "SmallC"

for building, en_name in BUILDING_MAP.items():
    
    # Folder
    binary_path = os.path.join(PROCESSED_FOLDER, f"{en_name}-timetable-binary")
    csv_files = glob.glob(os.path.join(binary_path, "*.csv"))
    
    building_all_days = []

    for file in tqdm(csv_files, desc=f"Aggregating {building}"):

        # Read in file
        df = pd.read_csv(file)
        date_str = os.path.basename(file).replace(".csv", "").split("_")[-1]
        date_obj = datetime.strptime(date_str, "%Y-%m-%d")
        
        # Calculate sum of inclass for each capacity group
        df['cap_group'] = df['capacity'].apply(classify_capacity)
        
        if 'room' not in df.columns and df.index.name == 'room':
            df = df.reset_index()

        time_cols = [c for c in df.columns if c not in ['room', 'capacity', 'cap_group']]
        df_melted = df.melt(id_vars=['room', 'cap_group'], value_vars=time_cols, var_name='Time', value_name='inclass')
        df_melted['inclass'] = pd.to_numeric(df_melted['inclass'], errors='coerce').fillna(0).astype(int)
        df_agg = df_melted.groupby(['Time', 'cap_group'])['inclass'].sum().unstack(fill_value=0)

        for col in ["BigC", "MediumC", "SmallC"]:
            if col not in df_agg.columns:
                df_agg[col] = 0
        
        # Add time info
        df_agg = df_agg.reset_index()
        df_agg['Date'] = date_str
        df_agg['DoW'] = date_obj.isoweekday() 
        
        building_all_days.append(df_agg)

    # Save data
    if building_all_days:
        
        final_df = pd.concat(building_all_days, ignore_index=True)
        final_df = final_df[['Date', 'DoW', 'Time', 'BigC', 'MediumC', 'SmallC']]
        
        ntu_time_order = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9', '10', 'A', 'B', 'C', 'D']
        final_df['Time'] = pd.Categorical(final_df['Time'], categories=ntu_time_order, ordered=True)
        final_df = final_df.sort_values(['Date', 'Time'])

        save_name = f"{en_name}_timetable_summary.csv"
        final_df.to_csv(os.path.join(PROCESSED_FOLDER, save_name), index=False)

print("Done!")