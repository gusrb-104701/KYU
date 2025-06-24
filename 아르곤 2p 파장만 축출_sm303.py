# -*- coding: utf-8 -*-
"""
Created on Tue Mar 26 16:31:23 2024

@author: USER
"""

import pandas as pd
import os
import glob
# Get the current working directory
current_dir = os.getcwd()
# Define the base folder for the original folder and new folder
base_folder = os.path.join(current_dir, "paper data/Processed data-Map")

# Use glob to find all CSV files in all subdirectories of the 'Argon 2p selection' folder
csv_files1 = glob.glob(os.path.join(base_folder, '**', '*.csv'), recursive=True)
# CSV 파일 경로
base_dir = 'C:/Users/USER/Desktop/Codeset/Plasma Parameter 계산/paper data/Processed data-Map'

output_folder = os.path.join(base_dir, 'Processed data-Map argon 2p selection')
os.makedirs(output_folder, exist_ok=True)  # Create the output folder if it does




# 범위 지정
ranges = [
    # 범위들을 여기에 넣으세요.
    (750, 750.99), (667, 667.99), (826, 826.99), (772, 772.99), (727, 727.99),
    (696, 696.99), (840, 840.99), (738, 738.99), (706, 706.99), (851.5, 852.5),
    (794, 794.9), (746, 746.9), (714, 715), (856, 857), (751, 752),
    (922, 922.99), (801, 801.99), (763, 763.99), (935, 936), (867, 867.99),
    (810, 810.99), (772, 772.99), (979, 979.99), (842, 842.99), (801, 801.9),
    (811, 811.99)
]


# Loop through each CSV file in the '202403' folder
for csv_file_path in csv_files1:
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