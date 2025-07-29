# -*- coding: utf-8 -*-
"""
Created on Tue Jul 15 18:35:55 2025

@author: gusrb
"""

# 전체 파장을 조건별, 채널별로 비교하는 코드
# 그래프 비교 가능 및 수치 확인 가능

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# 1️⃣ 조건별 파일 경로 설정 (파일 경로만 바꾸면 바로 실행 가능!)
files_info = {
    '5 Torque':  r'C:\Users\gusrb\Desktop\WGS 실험 DATA\20250612 sample test, Center Edge Test\120_5torque Edge_trend확인\141017\20250530-141017_8.csv',
    '10 Torque':  r'C:\Users\gusrb\Desktop\WGS 실험 DATA\20250612 sample test, Center Edge Test\121 10torque edge_trend확인\142722\20250530-142722_8.csv',
    '15 Torque': r'C:\Users\gusrb\Desktop\WGS 실험 DATA\20250612 sample test, Center Edge Test\122 15torque edge_trend확인\143651\20250530-143651_8.csv'
}

skip_header = 5
intensity_dict = {}
wavelengths = None
threshold = 100  # 안정화 기준 intensity

# 2️⃣ 안정화 구간 자동 탐지 후 평균 intensity 계산
for label, path in files_info.items():
    df = pd.read_csv(path, header=None, skiprows=skip_header)
    if wavelengths is None:
        wavelengths = df.iloc[0, 8:].astype(float).values
    data = df.iloc[1:, 8:].astype(float).reset_index(drop=True)

    # 자동 start/end 구간 찾기
    start, end = None, None
    for i in range(len(data)):
        row = data.iloc[i]
        if start is None and (row > threshold).any():
            start = i
        elif start is not None and not (row > threshold).any():
            end = i
            break
    if start is None: start = 0
    if end is None: end = len(data)

    stable_data = data.iloc[start:end]
    mean_intensity = stable_data.mean(axis=0).values
    intensity_dict[label] = mean_intensity

# 3️⃣ 평균 intensity DataFrame 생성
df_intensity = pd.DataFrame(intensity_dict, index=wavelengths)

# 4️⃣ 조건별로 5 < 10 < 15인 파장 필터링
increasing = df_intensity[
    (df_intensity['5 Torque'] > df_intensity['10 Torque']) &
    (df_intensity['10 Torque'] > df_intensity['15 Torque'])
]

# 5️⃣ 차이(range) 계산 및 상위 30개 추출
increasing['range'] = increasing.max(axis=1) - increasing.min(axis=1)
top_n = 30
top_wavelengths = increasing.sort_values(by='range', ascending=False).head(top_n)

# 6️⃣ 콘솔 출력
print(f"\n조건별 intensity 차이가 큰 상위 {top_n}개 파장:")
print(top_wavelengths[['5 Torque', '10 Torque', '15 Torque', 'range']])

# 7️⃣ 시각화 (파장 오름차순 정렬)
df_top30 = df_intensity.loc[top_wavelengths.index].sort_index()
x_labels = df_top30.index.astype(str)
x = np.arange(len(x_labels))
width = 0.25

plt.figure(figsize=(14, 6))
plt.bar(x - width, df_top30['5 Torque'], width, label='5 Torque', color='blue')
plt.bar(x,         df_top30['10 Torque'], width, label='10 Torque', color='orange')
plt.bar(x + width, df_top30['15 Torque'], width, label='15 Torque', color='green')

plt.xticks(x, x_labels, rotation=45)
plt.xlabel('Wavelength (nm)')
plt.ylabel('Mean Intensity (a.u.)')
plt.title('Top 30 Wavelengths: Grouped Bar Chart (5 > 10 > 15 Torque)')
plt.grid(axis='y')
plt.legend()
plt.tight_layout()
plt.show()