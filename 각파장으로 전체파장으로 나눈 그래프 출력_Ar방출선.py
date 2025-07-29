# -*- coding: utf-8 -*-
"""
Created on Wed Jul 23 18:24:01 2025

@author: gusrb
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from typing import Dict

# ✅ 중성 Ar I의 대표 방출선 목록 (600~850 nm)
ar_wavelengths = np.array([
    696.54, 706.72, 714.70, 727.29, 738.40,
    750.39, 763.51, 772.42, 794.82, 800.62,
    811.53, 826.45
])

# 1️⃣ 안정화 구간 탐지 함수
def detect_stable_region(df: pd.DataFrame, intensity_threshold: float = 100.0):
    mask = (df > intensity_threshold).sum(axis=1) > 0
    start_idx = mask.idxmax() if mask.any() else 0
    end_idx = (~mask).idxmax() if (~mask).any() and (~mask).idxmax() > start_idx else len(mask) - 1
    return start_idx, end_idx

# 2️⃣ Ar 방출선만 시각화하는 메인 함수
def plot_known_ar_lines(file_dict: Dict[str, str],
                        intensity_threshold: float = 100.0,
                        skip_columns: int = 8,
                        match_tol: float = 0.2):  # 허용 오차 ±0.2nm
    all_mean_intensities = {}
    wavelengths = None

    # 2-1. 안정화 구간에서 평균 intensity 추출
    for label, path in file_dict.items():
        df_raw = pd.read_csv(path, header=None, skiprows=5)
        if wavelengths is None:
            wavelengths = df_raw.iloc[0, skip_columns:].astype(float).values
        data = df_raw.iloc[1:, skip_columns:].astype(float).reset_index(drop=True)

        start_idx, end_idx = detect_stable_region(data, intensity_threshold)
        stable_data = data.iloc[start_idx:end_idx+1]
        mean_intensities = stable_data.mean(axis=0)
        all_mean_intensities[label] = mean_intensities

    # 2-2. Ar 방출선 중 실제 데이터에 존재하는 파장 찾기 (±0.2nm 오차 허용)
    matched_ar_waves = []
    matched_indices = []
    for ar_wave in ar_wavelengths:
        idx = np.where(np.isclose(wavelengths, ar_wave, atol=match_tol))[0]
        if len(idx) > 0:
            matched_ar_waves.append(wavelengths[idx[0]])
            matched_indices.append(idx[0])

    # 2-3. 일치한 Ar 파장 출력
    print(f"\n🔍 데이터에 존재하는 Ar 방출선 ({len(matched_ar_waves)}개): {matched_ar_waves}")

    # 2-4. 각 Ar 파장 기준으로 log(Intensity / Base) 그래프 출력
    for wave, idx in zip(matched_ar_waves, matched_indices):
        plt.figure(figsize=(10, 5))
        for label in file_dict:
            intensities = all_mean_intensities[label]
            base_val = intensities[idx]
            log_ratio = np.log10(intensities / base_val)
            plt.plot(wavelengths, log_ratio, label=label)

        plt.title(f"Ar Line {wave:.2f} nm 기준 log(Intensity / Base)")
        plt.xlabel("Wavelength (nm)")
        plt.ylabel("log10(Intensity / {base:.2f})")
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.show()

# 3️⃣ 사용자 PC 환경에 맞는 CSV 경로 설정
file_paths = {
    "5torque": r"C:\Users\gusrb\Desktop\WGS 실험 DATA\20270704 Ar 110, Pressure 50, Source or Bias 200\20250704 Bias Test 5_1\20250704-140935_1.csv",
    "10torque": r"C:\Users\gusrb\Desktop\WGS 실험 DATA\20270704 Ar 110, Pressure 50, Source or Bias 200\20250704 Bias Test10_1\20250704-134104_1.csv",
    "15torque": r"C:\Users\gusrb\Desktop\WGS 실험 DATA\20270704 Ar 110, Pressure 50, Source or Bias 200\20250704 Bias Test 15_1\20250704-114048_1.csv"
}

# 4️⃣ 실행
plot_known_ar_lines(file_paths, intensity_threshold=100.0)