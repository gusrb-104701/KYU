import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from typing import Dict

# 1️⃣ 안정화 구간 탐지 함수
def detect_stable_region(df: pd.DataFrame, intensity_threshold: float = 100.0):
    mask = (df > intensity_threshold).sum(axis=1) > 0
    start_idx = mask.idxmax() if mask.any() else 0
    end_idx = (~mask).idxmax() if (~mask).any() and (~mask).idxmax() > start_idx else len(mask) - 1
    return start_idx, end_idx

# 2️⃣ 분석 및 시각화 함수
def analyze_and_plot_each_top_wavelength(file_dict: Dict[str, str],
                                         intensity_threshold: float = 100.0,
                                         top_n: int = 50,
                                         skip_columns: int = 8):
    all_mean_intensities = {}
    wavelengths = None

    # 각 조건별 안정화 구간에서 파장별 평균 intensity 계산
    for label, path in file_dict.items():
        df_raw = pd.read_csv(path, header=None, skiprows=5)
        if wavelengths is None:
            wavelengths = df_raw.iloc[0, skip_columns:].astype(float).values
        data = df_raw.iloc[1:, skip_columns:].astype(float).reset_index(drop=True)

        start_idx, end_idx = detect_stable_region(data, intensity_threshold)
        stable_data = data.iloc[start_idx:end_idx+1]
        mean_intensities = stable_data.mean(axis=0)
        all_mean_intensities[label] = mean_intensities

    # 각 파장을 기준으로 전체 파장을 나눈 log 비율 계산 후 조건별 차이 측정
    ratio_diffs = pd.Series(index=wavelengths, dtype=float)

    for i, base_wave in enumerate(wavelengths):
        diffs = []
        for label in file_dict:
            base_val = all_mean_intensities[label].iloc[i]
            if base_val == 0:
                diffs.append(np.nan)
            else:
                ratio = np.log10(all_mean_intensities[label] / base_val)
                diffs.append(ratio)

        if pd.isna(diffs).any():  # <-- 여기 수정 완료
            ratio_diffs.iloc[i] = np.nan
        else:
            stacked = pd.DataFrame(diffs).T
            ratio_diffs.iloc[i] = stacked.max(axis=1).iloc[0] - stacked.min(axis=1).iloc[0]

    # 조건별 차이 큰 상위 top_n 파장 선택
    top_waves = ratio_diffs.sort_values(ascending=False).dropna().head(top_n).index

    print("📌 조건별 log10 비율 차이가 큰 상위 50개 파장:")
    for wave in top_waves:
        print(f"{wave:.2f} nm\t➤ 차이: {ratio_diffs[wave]:.5f}")

    # 각 상위 파장을 기준으로 그래프 개별 출력
    for base_wave in top_waves:
        base_idx = np.where(wavelengths == base_wave)[0][0]
        plt.figure(figsize=(10, 5))
        for label in file_dict:
            intensities = all_mean_intensities[label]
            base_val = intensities[base_idx]
            log_ratio = np.log10(intensities / base_val)
            plt.plot(wavelengths, log_ratio, label=label)

        plt.title(f"All Wavelengths / {base_wave:.2f} nm (log10 Scale)")
        plt.xlabel("Wavelength (nm)")
        plt.ylabel("log10(Intensity / Base Wavelength Intensity)")
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.show()

# 3️⃣ 파일 경로 정의 (여기 반드시 사용자 환경에 맞게 수정!)
file_paths = {
 "1torque": r"C:\Users\gusrb\Desktop\WGS 실험 DATA\20250808 분석 data\Ar 110 Source 1200 integration time10 total\Ar 110 Source 1200 integration time10 01_1.csv",
 "5torque": r"C:\Users\gusrb\Desktop\WGS 실험 DATA\20250808 분석 data\Ar 110 Source 1200 integration time10 total\Ar 110 Source 1200 integration time10 05_1.csv",
 "10torque": r"C:\Users\gusrb\Desktop\WGS 실험 DATA\20250808 분석 data\Ar 110 Source 1200 integration time10 total\Ar 110 Source 1200 integration time10 10_1.csv",
 "15torque": r"C:\Users\gusrb\Desktop\WGS 실험 DATA\20250808 분석 data\Ar 110 Source 1200 integration time10 total\Ar 110 Source 1200 integration time10 15_1.csv"
}

# 4️⃣ 실행
analyze_and_plot_each_top_wavelength(file_paths, intensity_threshold=100.0, top_n=50)