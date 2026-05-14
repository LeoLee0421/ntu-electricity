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


# Set paths ------------------------------------------------------------------------------------
WORK_DIR = os.getcwd()
RAW_FOLDER = os.path.join(WORK_DIR, "data", "raw")
PROCESSED_FOLDER = os.path.join(WORK_DIR, "data", "processed")

# 建築對應（中文 → 英文）
BUILDING_MAP = {
    "普通": "putong",
    #"綜合": "zonghe",
    #"博雅": "boya",
    "新生": "xinsheng",
    "共同": "gongtong"
}


# Readin data ------------------------------------------------------------------------------------
electricity_data = pd.read_csv(os.path.join(PROCESSED_FOLDER, "target_electricity_dataframe.csv"))
weather_data = pd.read_csv(os.path.join(PROCESSED_FOLDER, "weather_dataframe.csv"))
weather_data['DateTime'] = pd.to_datetime(weather_data['DateTime'])


# Electricity cleaning ------------------------------------------------------------------------------------
electricity_data['DateTime'] = pd.to_datetime(electricity_data['DateTime'])
electricity_data['DoW'] = electricity_data['DateTime'].dt.dayofweek + 1  
electricity_data['Hour_Str'] = electricity_data['DateTime'].dt.strftime('%H:00')

hour_to_period = {
    "08:00": "1", "09:00": "2", "10:00": "3", "11:00": "4", 
    "12:00": "5", "13:00": "6", "14:00": "7", "15:00": "8", 
    "16:00": "9", "17:00": "10", "18:00": "A", "19:00": "B", 
    "20:00": "C", "21:00": "D", "06:00" : "Basic06", "07:00" : "Basic07"
}

electricity_data['Period'] = electricity_data['Hour_Str'].map(hour_to_period)
electricity_data.loc[electricity_data['DoW'] > 5, 'Period'] = np.nan


# ECombine dataset ------------------------------------------------------------------------------------

for zh_name, en_name in BUILDING_MAP.items():

    print(f"正在整合 {en_name} 的數據...")

    # Classroom data
    schedule_path = os.path.join(PROCESSED_FOLDER, "timetable-summary", f"{en_name}_timetable_summary.csv")
    schedule_df = pd.read_csv(schedule_path)
    schedule_df['Time'] = schedule_df['Time'].astype(str)
    
    # Electricity data
    df_building = electricity_data[['DateTime', 'DoW', 'Period', en_name]].copy()

    # Climate data
    df_building = pd.merge(df_building, weather_data, on='DateTime', how='left')
    df_building['Temp'] = df_building['Temp'].interpolate()
    
    # Merge classroom
    schedule_df['Date'] = pd.to_datetime(schedule_df['Date']).dt.strftime('%Y-%m-%d')
    df_building['Date_Str'] = df_building['DateTime'].dt.strftime('%Y-%m-%d')
    #df_building = df_building.dropna(subset=["Period"])
    
    df_merged = pd.merge(
        df_building, 
        schedule_df[['Date', 'Time', 'BigC', 'SmallC']], 
        left_on=['Date_Str', 'Period'], 
        right_on=['Date', 'Time'], 
        how='left'
    ).drop(columns=['Date', 'Date_Str', 'Time'])

    # Post-processing
    df_merged[['BigC', 'SmallC']] = df_merged[['BigC', 'SmallC']].fillna(0)
    df_merged = df_merged.rename(columns={en_name: 'Electricity'})
    df_merged = df_merged.dropna(subset=['Electricity'])
    
    # Save file
    output_filename = f"{en_name}_final_combined_alltime.csv"
    df_merged.to_csv(os.path.join(PROCESSED_FOLDER, "alltime", output_filename), index=False)
    print(f"Save at {output_filename}")

print("Done!")