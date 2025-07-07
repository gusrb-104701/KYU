# -*- coding: utf-8 -*-
"""
Created on Mon Jul  7 18:01:11 2025

@author: gusrb
"""


# DH ksp DATA 처리 코드-그래프추가 실행 후 생성된 LINE RATIO 파일에 대해 시각화 진행

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
import glob

# 1. 대상 폴더 설정
folder = r"C:/Users/gusrb/Desktop/WGS 실험 DATA/20250704 분석data/line ratio_811.59_750.49"

# 2. glob 사용해서 하위 폴더 포함한 모든 *_line ratio.csv 파일 불러오기
file_list = glob.glob(os.path.join(folder, "**/*_line ratio.csv"), recursive=True)
file_list.sort()  # 이름순 정렬 (옵션)

# 3. 파일 이름에서 조건명(Bias 5, 10, 15 등) 추출해서 label 리스트 생성
file_labels = [os.path.basename(f).split("Bias Test ")[-1].split("_")[0] + " torque" for f in file_list]

# 4. 안정화 구간 행 index
stable_start = 10
stable_end = 21
time_axis = np.arange(stable_end - stable_start)

# ✅ 조건별 색상 지정
color_dict = {
    "05 torque": "r",
    "10 torque": "g",
    "15 torque": "b"
}



# 5. CH1~CH8 별 그래프 생성
for ch in range(8):
    plt.figure(figsize=(10, 5))
    for filepath, label in zip(file_list, file_labels):
        df = pd.read_csv(filepath, header=None)
        if df.shape[0] < stable_end:
            print(f"⚠️ {filepath} 파일의 행 수가 안정화구간보다 작습니다.")
            continue
        ch_data = df.iloc[stable_start:stable_end, ch]
        
        # ✅ Scale 변경
        safe_data = ch_data.clip(lower=1e-6)
        converted = -1 / np.log(safe_data) / 2
        
        plt.plot(time_axis, converted, label=label, color=color_dict.get(label, 'k'))
    
    plt.title(f"Channel {ch+1} - Stable Region Line Ratio Comparison")
    plt.xlabel("Time Index (Row 10 to 20)")
    plt.ylabel("Line Ratio")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    save_dir = r"C:/Users/gusrb/Desktop/그래프결과"  # 원하는 저장 경로로 바꿔도 됨
    os.makedirs(save_dir, exist_ok=True)  # 폴더 없으면 자동 생성
    save_path = os.path.join(save_dir, f"line_ratio_CH{ch+1}.png") #그래프추가
    plt.savefig(save_path, dpi=300) # 그래프 해상도
    plt.show()