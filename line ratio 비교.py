# -*- coding: utf-8 -*-
"""
Created on Mon Jun 30 16:48:20 2025

@author: gusrb
"""

# 파장 선정 후 line ratio 계산 및 파일생성 (channel 구분 O)
# SYPDER 폴더에 저장

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os

# 경로설정
folder = r"C:/Users/gusrb/Desktop/WGS 실험 DATA/20250704 분석data"

# Channel 통합시킨 대상파일 list
filenames = [
    "20250704 Bias Test 15_1.csv",
    "20250704 Bias Test 15_2.csv",
    "20250704 Bias Test 15_3.csv",
    "20250704 Bias Test 10_1.csv",
    "20250704 Bias Test 10_2.csv",
    "20250704 Bias Test 10_3.csv",
    "20250704 Bias Test 05_1.csv",
    "20250704 Bias Test 05_2.csv",
    "20250704 Bias Test 05_3.csv",
    "20250704 Source Test 15_1.csv",
    "20250704 Source Test 15_2.csv",
    "20250704 Source Test 15_3.csv",
    "20250704 Source Test 10_1.csv",
    "20250704 Source Test 10_2.csv",
    "20250704 Source Test 10_3.csv",
    "20250704 Source Test 05_1.csv",
    "20250704 Source Test 05_2.csv",
    "20250704 Source Test 05_3.csv"
]

 #file names 정상저장 확인
print("filename check")
print(filenames)

# 파일별 색상 (red, green, blue)
colors = ['r', 'g', 'b']  

plt.figure(figsize=(14, 6))  # 전체 그래프 , polt 생성

# 2. 파일별 반복
for idx, file in enumerate(filenames):
    filepath = os.path.join(folder, file)
    df = pd.read_csv(filepath, header=0)

    # 컬럼 이름이 전부 같기 때문에 이름 중복 제거
    new_columns = ['Time'] + [f"{col}_{i}/8" for i, col in enumerate(df.columns[1:])]
    df.columns = new_columns

    # 750.49 / 751.47 리스트 구분  > 파장 selection후 수정작업 필요
    cols_750 = [col for col in df.columns if '811.59' in col] #ppt에 정리한 파장기준으로 다시 뽑아봐야함
    cols_751 = [col for col in df.columns if '750.49' in col]

    # 예외 처리
    if len(cols_750) != len(cols_751):
        raise ValueError(f"{file} → 750.49nm 개수: {len(cols_750)}, 751.47nm 개수: {len(cols_751)}")
    
    # line-ratio에 빈 행렬 설정 (추후 값들을 옆으로 쌓아줄 것)
    line_ratio = np.empty((len(df[cols_750[0]]),0))

    # 채널별 line ratio 계산 및 시각화
    for i in range(len(cols_750)):
        # np.c_ = np.concatenate(, axis=1) 축 1방향 (오른쪽 방향 = 
        temp_ratio = df[cols_750[i]] / df[cols_751[i]]
        line_ratio = np.c_[line_ratio, temp_ratio ]
        label = f'{file[:-4]}_ch{i+1}'
        plt.plot(df['Time'], temp_ratio, label=label, color=colors[idx% len(colors)], alpha=0.5)
    np.savetxt(f"{file}_line ratio.csv",line_ratio,delimiter=',')

"""
# 3. 그래프 설정
plt.xlabel('Time (s)')
plt.ylabel('Line Ratio (750.49 / 751.47)')
plt.title('Line Ratio by Channel across All Files')
plt.grid(True)
plt.legend(fontsize=8, ncol=2)
plt.tight_layout()
plt.show()
"""
