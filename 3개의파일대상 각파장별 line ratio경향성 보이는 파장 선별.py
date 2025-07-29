import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
import math

# 🔧 실험 설정
file_paths = [
    r"C:\Users\gusrb\Desktop\WGS 실험 DATA\20250612 sample test, Center Edge Test\120_5torque Edge_trend확인\141017\20250530-141017_8.csv",  # 5 Torque
    r"C:\Users\gusrb\Desktop\WGS 실험 DATA\20250612 sample test, Center Edge Test\121 10torque edge_trend확인\142722\20250530-142722_8.csv",  # 10 Torque
    r"C:\Users\gusrb\Desktop\WGS 실험 DATA\20250612 sample test, Center Edge Test\122 15torque edge_trend확인\143651\20250530-143651_8.csv",  # 15 Torque
]
stable_ranges = [(12, 34), (10, 33), (16, 39)] #변경필요
labels = ["5 Torque", "10 Torque", "15 Torque"]
header_skip = 5
min_wavelength = 128.55
top_n_intensity = 50
invalid_ranges = [(777.03, 777.03), (749.51, 750.49), (428.61, 428.61),
                  (703.57, 703.57), (656.03, 656.03), (306.72, 306.72)]

# 함수 정의
def is_valid_wavelength(wl):
    try:
        wl = float(wl)
        if wl < min_wavelength:
            return False
        for start, end in invalid_ranges:
            if start <= wl <= end:
                return False
        return True
    except:
        return False

def extract_valid_wavelengths(header_row):
    wls, idxs = [], []
    for i, val in enumerate(header_row[1:]):
        if is_valid_wavelength(val):
            wls.append(float(val))
            idxs.append(i + 1)
    return np.array(wls), idxs

# 데이터 로딩
intensity_data = {}
wavelengths, valid_indices = None, []

for path, label, srange in zip(file_paths, labels, stable_ranges):
    df = pd.read_csv(path, header=None, skiprows=header_skip)
    if wavelengths is None:
        wavelengths, valid_indices = extract_valid_wavelengths(df.iloc[0].values)
    stable_df = df.iloc[srange[0]:srange[1], valid_indices].astype(float)
    intensity_data[label] = stable_df.mean(axis=0).values

# 상위 intensity 파장 기준 추출
top_indices = np.argsort(intensity_data["5 Torque"])[-top_n_intensity:]
top_wavelengths = wavelengths[top_indices]

# 라인 레이쇼 계산
torque_order = ["5 Torque", "10 Torque", "15 Torque"]
trend_pairs, trend_data = [], []

for i, ref_idx in enumerate(top_indices):
    ref_wl = wavelengths[ref_idx]
    for j, tgt_idx in enumerate(top_indices):
        if ref_idx == tgt_idx:
            continue
        tgt_wl = wavelengths[tgt_idx]
        values = []
        for torque in torque_order:
            ref_val = intensity_data[torque][ref_idx]
            tgt_val = intensity_data[torque][tgt_idx]
            ratio = tgt_val / ref_val if ref_val != 0 else np.nan
            values.append(ratio)
        if np.all(np.diff(values) > 0) or np.all(np.diff(values) < 0):
            trend_pairs.append((ref_wl, tgt_wl))
            trend_data.append(values)

# Reference 별로 데이터 정리
unique_refs = sorted(set(ref for ref, _ in trend_pairs))
ref_to_trend = {ref: [] for ref in unique_refs}
ref_to_data = {ref: [] for ref in unique_refs}
for (ref, tgt), ratios in zip(trend_pairs, trend_data):
    ref_to_trend[ref].append(tgt)
    ref_to_data[ref].append(ratios)

# 📊 시각화: Reference 파장 별 subplot, x축 = 분자 파장, y축 = 라인 레이쇼
fig_cols = 3
fig_rows = math.ceil(len(unique_refs) / fig_cols)
fig, axes = plt.subplots(fig_rows, fig_cols, figsize=(fig_cols * 6, fig_rows * 4), squeeze=False)

bar_width = 0.2
x_tick_offset = [-bar_width, 0, bar_width]

for i, ref in enumerate(unique_refs):
    row, col = divmod(i, fig_cols)
    ax = axes[row][col]
    targets = ref_to_trend[ref]
    data = ref_to_data[ref]
    x_positions = np.arange(len(targets))

    for j, torque_label in enumerate(torque_order):
        y_values = [ratios[j] for ratios in data]
        ax.bar(x_positions + x_tick_offset[j], y_values, width=bar_width, label=torque_label)

    ax.set_xticks(x_positions)
    ax.set_xticklabels([f"{tgt:.1f}" for tgt in targets], rotation=45)
    ax.set_title(f"Reference: {ref:.2f} nm")
    ax.set_xlabel("Target Wavelength (Numerator)")
    ax.set_ylabel("Line Ratio")
    ax.grid(True)
    ax.legend()

# 빈 subplot 제거
for j in range(len(unique_refs), fig_rows * fig_cols):
    fig.delaxes(axes[j // fig_cols][j % fig_cols])

plt.tight_layout()
plt.show()