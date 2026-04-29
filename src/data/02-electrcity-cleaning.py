import pandas as pd
import os
import numpy as np

FILE_FOLDER = "/Users/neptunelee/Desktop/NTU/環資/Github/data/electricity"

files = [
    "putong_electricity_01.html",
    "putong_electricity_02.html",
    "putong_electricity_03.html"
]

dfs = []

for file in files:
    
    df_html = pd.read_html(os.path.join(FILE_FOLDER, file))
    df_html = df_html[1]
    df_html.to_csv(os.path.join(FILE_FOLDER, f"{file[:21]}.csv"))

    df = pd.read_csv(
        os.path.join(FILE_FOLDER, f"{file[:21]}.csv"),
        skiprows = 1,
        index_col = "日期時間",
        parse_dates = True
    )
    
    df = df.drop(columns=["0"])
    df = df.apply(pd.to_numeric, errors="coerce")
    df.loc[df["用電度數"] < 0, "用電度數"] = pd.NA
    dfs.append(df)

# Aggreagate three dataframes by summing them up
df_sum = dfs[0].copy()
for df in dfs[1:]: df_sum = df_sum.add(df, fill_value=0)
df_sum = df_sum.sort_index()
df_sum = df_sum.round(2)

output_path = os.path.join(FILE_FOLDER, "putong_electricity_merged.csv")
df_sum.to_csv(output_path)

print("Saved to:", output_path)