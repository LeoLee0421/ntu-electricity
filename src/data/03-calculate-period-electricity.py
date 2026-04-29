import pandas as pd
import os

# Set path
WORK_DIR = os.getcwd()
DATA_DIR = os.path.join(WORK_DIR, "data", "electricity")
OUTPUT_DIR = os.path.join(WORK_DIR, "data", "cleandata")

# Read in data
df = pd.read_csv(os.path.join(DATA_DIR, "putong_electricity_merged.csv"))

# Set time range
df['日期時間'] = pd.to_datetime(df['日期時間'])
df = df[(df['日期時間'] >= '2026-02-23') & (df['日期時間'] <= '2026-04-04')]
df = df[df['日期時間'].dt.weekday < 5]
df['date'] = df['日期時間'].dt.date
df['time'] = df['日期時間'].dt.time

time_ranges = {
    "1": ("08:10", "09:00"),
    "2": ("09:10", "10:00"),
    "3": ("10:20", "11:10"),
    "4": ("11:20", "12:10"),
    "5": ("12:20", "13:10"),
    "6": ("13:20", "14:10"),
    "7": ("14:20", "15:10"),
    "8": ("15:30", "16:20"),
    "9": ("16:30", "17:20"),
    "10": ("17:30", "18:20"),
    "A": ("18:25", "19:15"),
    "B": ("19:20", "20:10"),
    "C": ("20:15", "21:05"),
    "D": ("21:10", "22:00"),
}

result = {}

for label, (start, end) in time_ranges.items():
    start = pd.to_datetime(start).time()
    end = pd.to_datetime(end).time()
    temp = df[(df['time'] >= start) & (df['time'] <= end)]
    grouped = temp.groupby('date')['用電度數'].sum()
    result[label] = grouped

result_df = pd.DataFrame(result)

final_df = result_df.T
final_df = final_df.sort_index(axis=1)
final_df = final_df.fillna(0)
final_df = final_df.round(2)

print(final_df)
final_df.to_csv(os.path.join(OUTPUT_DIR, "period_electricity_clean.csv"), index=True)