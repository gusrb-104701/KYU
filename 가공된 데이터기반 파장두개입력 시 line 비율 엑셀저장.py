# -*- coding: utf-8 -*-
"""
Created on Thu Jul 24 16:40:45 2025

@author: gusrb
"""


#채널별로 가공된 데이터를 소스로 사용
#각 실험별 3개씩의 데이터 입력가능
#각 데이터들에 대해 특정 파장들의 line ratio 비교가능

import pandas as pd
import numpy as np

# ===================== 🔧 1. 설정 =====================
file_sets = {
    "Source Test1": {
        "5torque": r"C:\Users\gusrb\Desktop\WGS 실험 DATA\20250704 분석data\20250704 Source Test 05_1.csv",
        "10torque": r"C:\Users\gusrb\Desktop\WGS 실험 DATA\20250704 분석data\20250704 Source Test 10_1.csv",
        "15torque": r"C:\Users\gusrb\Desktop\WGS 실험 DATA\20250704 분석data\20250704 Source Test 15_1.csv",
    },
    "Source Test2": {
        "5torque": r"C:\Users\gusrb\Desktop\WGS 실험 DATA\20250704 분석data\20250704 Source Test 05_2.csv",
        "10torque": r"C:\Users\gusrb\Desktop\WGS 실험 DATA\20250704 분석data\20250704 Source Test 10_2.csv",
        "15torque": r"C:\Users\gusrb\Desktop\WGS 실험 DATA\20250704 분석data\20250704 Source Test 15_2.csv",
    },
    "Source Test3": {
        "5torque": r"C:\Users\gusrb\Desktop\WGS 실험 DATA\20250704 분석data\20250704 Source Test 05_3.csv",
        "10torque": r"C:\Users\gusrb\Desktop\WGS 실험 DATA\20250704 분석data\20250704 Source Test 10_3.csv",
        "15torque": r"C:\Users\gusrb\Desktop\WGS 실험 DATA\20250704 분석data\20250704 Source Test 15_3.csv",
    },
}

numerator_wavelength = 696.76 #분자입력
denominator_wavelength = 811.59 #분모입력


# ===================== 🔁 2. 처리 함수 =====================
def process_file_group(group_name, file_dict, num_wl, den_wl):
    result_df = pd.DataFrame()

    for torque_label, file_path in file_dict.items():
        df = pd.read_csv(file_path, header=None)
        wavelengths = df.iloc[0, 1:].astype(float).values
        data = df.iloc[1:, 1:].astype(float).reset_index(drop=True)

        result = {
            "Log": [],
            "Ratio": [],
            str(num_wl): [],
            str(den_wl): []
        }

        num_indices = np.where(wavelengths == num_wl)[0]
        den_indices = np.where(wavelengths == den_wl)[0]

        for ch in range(8):
            num_col = num_indices[ch]
            den_col = den_indices[ch]

            num_vals = data.iloc[:, num_col]
            den_vals = data.iloc[:, den_col]

            # 안정화 구간만 포함된 파일이므로 전체 평균 사용
            num_avg = num_vals.mean()
            den_avg = den_vals.mean()
            ratio = num_avg / den_avg
            log_val = np.log10(ratio)

            result["Log"].append(log_val)
            result["Ratio"].append(ratio)
            result[str(num_wl)].append(num_avg)
            result[str(den_wl)].append(den_avg)

        section_df = pd.DataFrame(result, index=[f"channel{i}" for i in range(1, 9)]).T.reset_index()
        section_df = section_df.rename(columns={"index": "Channel"})
        header_row = pd.DataFrame([[torque_label] + [""] * (section_df.shape[1] - 1)], columns=section_df.columns)

        result_df = pd.concat([result_df, header_row, section_df], ignore_index=True)

    return result_df


# ===================== 💾 3. 실행 + 수직 결합 + 엑셀 저장 =====================

final_df = pd.DataFrame()

for group_name, file_group in file_sets.items():
    df = process_file_group(group_name, file_group, numerator_wavelength, denominator_wavelength)

    # 실험명 한 줄 추가
    header = pd.DataFrame([[group_name] + [""] * (df.shape[1] - 1)], columns=df.columns)

    # 구분선 삽입
    final_df = pd.concat([final_df, header, df], ignore_index=True)

# 저장
output_path = fr"C:\Users\gusrb\Desktop\WGS 실험 DATA\20250704 분석data\{numerator_wavelength}_{denominator_wavelength}OES_Ratio_Summary_Vertical.xlsx"
with pd.ExcelWriter(output_path, engine='xlsxwriter') as writer:
    final_df.to_excel(writer, index=False, sheet_name="Summary")

print(f"✅ 엑셀 저장 완료: {output_path}")