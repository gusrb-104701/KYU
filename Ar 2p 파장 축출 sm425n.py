# -*- coding: utf-8 -*-
"""
Created on Tue Mar 26 16:31:23 2024

@author: USER
"""

import pandas as pd
import os
import glob

# CSV 파일 경로
base_dir = 'C:/Users/USER/Desktop/Codeset/높이에 따른 라디칼  및 epd'
csv_folder_path = os.path.join(base_dir, 'Raw data', '*.csv')
output_folder = os.path.join(base_dir, 'for radical')
os.makedirs(output_folder, exist_ok=True)  # Create the output folder if it does




# 범위 지정
ranges = [
    # 범위들을 여기에 넣으세요.
    (750.2, 750.6), (667.6, 668), (826.2, 826.6), (772.2, 772.5), (727.2, 727.5),
    (696.6, 696.9), (840.5, 840.7), (738.3, 738.5), (706.6, 706.8), (851.9, 852.1),
    (794.7, 794.9), (746.2, 746.5), (714.8, 715), (856.7, 856.9), (751.3, 751.5),
    (922.7, 922.9), (801.1, 801.3), (763.3, 763.5), (935.8, 936), (867.1, 867.4),
    (810.1, 810.3), (772.2, 772.4), (979.5, 979.7), (842.3, 842.5), (801.1, 801.3),
    (811.2, 811.5)
]


# Loop through each CSV file in the '202403' folder
for csv_file_path in glob.glob(csv_folder_path):
    df = pd.read_csv(csv_file_path, header=None)

    # Data processing operations
    df.drop(0, axis=1, inplace=True)
    df.iloc[0] = pd.to_numeric(df.iloc[0], errors='coerce')

    columns_to_include = []
    for start, end in ranges:
        columns_in_range = df.columns[1:][df.iloc[0, 1:].between(start, end)]
        sum_values = df[columns_in_range].iloc[1:].sum()
        if not sum_values.empty:
            max_sum_index = sum_values.idxmax()
            if 0 <= max_sum_index < len(df.columns):
                max_sum_value = df.iloc[0, max_sum_index-1]
                columns_to_include.extend(df.columns[1:][df.iloc[0, 1:] == max_sum_value])

    selected_data = df[columns_to_include].drop_duplicates()

    # Save the processed DataFrame to a new file in the 'Argon 2p selection' folder
    original_file_name = os.path.splitext(os.path.basename(csv_file_path))[0]
    output_file_path = os.path.join(output_folder, f'{original_file_name}_selected.csv')
    selected_data.to_csv(output_file_path, index=False)

    print(f"Processed and saved: '{output_file_path}'")

print("All files have been processed and saved.")