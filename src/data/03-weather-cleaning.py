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

# Main ----------------------------------------------------------------------------------------
weather_data_2024 = pd.read_csv(os.path.join(RAW_FOLDER, "climate_data_2024.csv"))
weather_data_2025 = pd.read_csv(os.path.join(RAW_FOLDER, "climate_data_2025.csv"))
weather_data = pd.concat([weather_data_2024, weather_data_2025], ignore_index=True)

weather_data = weather_data.rename(columns={'Unnamed: 0': 'DateTime', 'Tx': 'Temp'})
weather_data['DateTime'] = pd.to_datetime(weather_data['DateTime'])
weather_data = weather_data[['DateTime', 'Temp']]

weather_data.to_csv(os.path.join(PROCESSED_FOLDER, "weather_dataframe.csv"), index=False)

print('Save to weather_dataframe.csv successfully!')