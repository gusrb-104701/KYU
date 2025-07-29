# -*- coding: utf-8 -*-
"""
Created on Tue Jul 15 13:57:34 2025

@author: gusrb
"""

import pandas as pd
import matplotlib.pyplot as plt


# 조건별 파일 경로 및 안정화 구간 설정 (※ start/end는 "5행 제거 이후 기준 인덱스"임)
files_info = {
    '5 Torque': {
        'path': r'C:\Users\gusrb\Desktop\WGS 실험 DATA\20270704 Ar 110, Pressure 50, Source or Bias 200\20250704 Source Test 5_1\20250704-135945_8.csv',
        'start': 7,#안정화 row -6
        'end': 29  #안정화 row -5
    },
    '10 Torque': {
        'path': r'C:\Users\gusrb\Desktop\WGS 실험 DATA\20270704 Ar 110, Pressure 50, Source or Bias 200\20250704 Source Test 10_1\20250704-120548_8.csv',
        'start': 5,#안정화 row -6
        'end': 27  #안정화 row -5
    },
    '15 Torque': {
        'path': r'C:\Users\gusrb\Desktop\WGS 실험 DATA\20270704 Ar 110, Pressure 50, Source or Bias 200\20250704 Source Test 15_1\20250704-113410_8.csv',
        'start': 11 , #안정화 row -6
        'end': 34   #안정화 row -5
    }
}

skip_header = 5
intensity_dict = {}
wavelengths = None

# 1. 조건별 평균 intensity 계산
for label, info in files_info.items():
    df = pd.read_csv(info['path'], header=None, skiprows=skip_header)
    if wavelengths is None:
        wavelengths = df.iloc[0, 8:].astype(float).values
    ch_data = df.iloc[info['start']:info['end'], 8:].astype(float)
    mean_intensity = ch_data.mean(axis=0).values
    intensity_dict[label] = mean_intensity

# 2. DataFrame 생성 (index = 파장)
df_intensity = pd.DataFrame(intensity_dict, index=wavelengths)

# 3. 조건별로 5 < 10 < 15를 만족하는 파장만 필터링
increasing = df_intensity[
    (df_intensity['5 Torque'] < df_intensity['10 Torque']) &
    (df_intensity['10 Torque'] < df_intensity['15 Torque'])
]

# 4. 각 파장에서 조건별 intensity 차이 계산
increasing['range'] = increasing.max(axis=1) - increasing.min(axis=1)

# 5. 상위 N개 파장 출력
top_n = 30
top_wavelengths = increasing.sort_values(by='range', ascending=False).head(top_n)

print(f"\n조건별 intensity 차이가 큰 상위 {top_n}개 파장:")
print(top_wavelengths[['5 Torque', '10 Torque', '15 Torque', 'range']])


"""
# --- (1) 전체 increasing 파장 기준 라인+포인트 그래프 ---
plt.figure(figsize=(14, 6))
for cond in ['5 Torque', '10 Torque', '15 Torque']:
    plt.plot(increasing.index, increasing[cond], marker='o', linestyle='-', label=cond)

plt.xlabel('Wavelength (nm)')
plt.ylabel('Mean Intensity (a.u.)')
plt.title('Mean Intensity at Wavelengths where 5 < 10 < 15 Torque')
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()
"""


# --- (2) Top 30 파장 기준 포인트-only 그래프 (카테고리형 X축으로 공백 제거) ---
df_top30 = df_intensity.loc[df_intensity.index.isin(top_wavelengths.index)]
x_labels = df_top30.index.astype(str)  # 문자열로 변환 (카테고리형)

plt.figure(figsize=(14, 6))

# 마커 스타일 지정
colors = {'5 Torque': 'blue', '10 Torque': 'orange', '15 Torque': 'green'}
markers = {'5 Torque': 'o', '10 Torque': 's', '15 Torque': '^'}

for cond in ['5 Torque', '10 Torque', '15 Torque']:
    plt.plot(x_labels, df_top30[cond],
             marker=markers[cond], linestyle='',
             color=colors[cond], label=cond)

    # ⬆️ 포인트 위에 파장값 텍스트로 표시
  #  for x_str, y in zip(x_labels, df_top30[cond]):
   #     plt.text(x_str, y + 10, x_str, ha='center', fontsize=8)

# ✅ 카테고리형 X축으로 공백 문제 해결
plt.xlabel('Wavelength (nm)')
plt.ylabel('Mean Intensity (a.u.)')
plt.title('Top 30 Wavelengths: Intensity Points (Categorical X-axis)')
plt.xticks(rotation=45)
plt.grid(True, axis='y')
plt.legend()
plt.tight_layout()
plt.show()

import numpy as np

df_top30 = df_intensity.loc[df_intensity.index.isin(top_wavelengths.index)]
x_labels = df_top30.index.astype(str)  # 문자열 파장
x = np.arange(len(x_labels))  # 위치 인덱스
width = 0.25  # 막대 너비

plt.figure(figsize=(14, 6))

plt.bar(x - width, df_top30['5 Torque'], width, label='5 Torque', color='blue')
plt.bar(x,         df_top30['10 Torque'], width, label='10 Torque', color='orange')
plt.bar(x + width, df_top30['15 Torque'], width, label='15 Torque', color='green')

plt.xticks(x, x_labels, rotation=45)
plt.xlabel('Wavelength (nm)')
plt.ylabel('Mean Intensity (a.u.)')
plt.title('Top 30 Wavelengths: Grouped Bar Chart (5 < 10 < 15 Torque)')
plt.grid(axis='y')
plt.legend()
plt.tight_layout()
plt.show()

