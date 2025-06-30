# -*- coding: utf-8 -*-
"""
Created on Mon Jun 30 16:48:20 2025

@author: gusrb
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os

# 1. 공통 설정
folder = r"C:\Users\gusrb\Desktop\WGS 실험 DATA"
filenames = [
    "Channel_Total_OES 8Chnnel_test5.csv",
    "Channel_Total_OES 8Chnnel_test10.csv",
    "Channel_Total_OES 8Chnnel_test15.csv"
]

colors = ['r', 'g', 'b']  # 파일별 색상 (red, green, blue)

plt.figure(figsize=(14, 6))  # 전체 그래프

# 2. 파일별 반복
for idx, file in enumerate(filenames):
    filepath = os.path.join(folder, file)
    df = pd.read_csv(filepath, header=0)

    # 컬럼 이름이 전부 같기 때문에 이름 중복 제거
    new_columns = ['Time'] + [f"{col}_{i}/8" for i, col in enumerate(df.columns[1:])]
    df.columns = new_columns

    # 750.49 / 751.47 리스트 구분
    cols_750 = [col for col in df.columns if '750.49' in col]
    cols_751 = [col for col in df.columns if '811.59' in col]

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
        plt.plot(df['Time'], temp_ratio, label=label, color=colors[idx], alpha=0.5)
    np.savetxt(f"{file}_line ratio.csv",line_ratio,delimiter=',')
        

# 3. 그래프 설정
plt.xlabel('Time (s)')
plt.ylabel('Line Ratio (750.49 / 751.47)')
plt.title('Line Ratio by Channel across All Files')
plt.grid(True)
plt.legend(fontsize=8, ncol=2)
plt.tight_layout()
plt.show()

