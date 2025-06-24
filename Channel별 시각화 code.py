# -*- coding: utf-8 -*-
"""
Created on Tue Jun 24 15:44:51 2025

@author: gusrb
"""

# -*- coding: utf-8 -*-
"""
3D OES Intensity Surface Plot
– 1~5행 메타데이터 건너뛰기
– 6행을 헤더(컬럼명)으로 사용
– X: 파장, Y: 시간(초), Z: intensity
"""

# -*- coding: utf-8 -*-
"""
3D OES Intensity Surface Plot
– 1~5행 메타데이터 건너뛰기
– 6행을 헤더로 사용
– 시간축은 실험 시작을 0초로 기준 설정
– X: 파장, Y: 시간(초), Z: intensity
"""

# -*- coding: utf-8 -*-
"""
3D OES Intensity Surface Plot (Axes Swapped)
– 1~5행 메타데이터 건너뛰기
– 6행을 헤더로 사용
– X축: 시간(초), Y축: 파장(nm), Z축: intensity
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # 3D 축 도구

# -------------------------------------------------------------------
# 1) CSV 파일 경로
file_path = r"C:\Users\gusrb\Desktop\WGS 실험 DATA\OES 8Chnnel_test15\20250530-143651_8.csv"

# -------------------------------------------------------------------
# 2) CSV 읽기: 상위 5행(skiprows=5) 건너뛰고, 6번째 줄(header=0)을 컬럼명으로 사용
df = pd.read_csv(file_path, skiprows=5, header=0)

# -------------------------------------------------------------------
# 3) 컬럼명 분리
raw_cols  = list(df.columns)
time_col  = raw_cols[0]    # 'Time'
wave_cols = raw_cols[1:]   # 파장 컬럼명들

# -------------------------------------------------------------------
# 4) 파장 이름을 숫자 리스트로 변환
wavelengths = []
for col in wave_cols:
    if "~" in col:
        num_str = col.split("~")[0].strip()
    else:
        num_str = col.strip()
    wavelengths.append(float(num_str))

# -------------------------------------------------------------------
# 5) 시간 문자열 → 초(sec) 변환 함수 정의
def to_seconds(tstr):
    parts = tstr.strip().split(":")
    sec = float(parts[-1])              # 초 부분
    if len(parts) >= 2:
        sec += int(parts[-2]) * 60      # 분 부분
    if len(parts) == 3:
        sec += int(parts[0]) * 3600     # 시 부분
    return sec

# 6) Time_sec 컬럼 생성 및 실험 시작 시점을 0초로 보정
df["Time_sec"] = df[time_col].map(to_seconds)
df["Time_sec"] -= df["Time_sec"].iloc[0]

# -------------------------------------------------------------------
# 7) intensity 매트릭스 준비
intensity = df[wave_cols].astype(float).values  # shape (T, W)

# -------------------------------------------------------------------
# 8) meshgrid 생성 (indexing='ij'로 time × wavelength 매핑)
#    X_mesh[i,j] = time[i], Y_mesh[i,j] = wavelength[j]
X_mesh, Y_mesh = np.meshgrid(df["Time_sec"].values,
                             wavelengths,
                             indexing='ij')

# -------------------------------------------------------------------
# 9) 3D Surface Plot 그리기
fig = plt.figure(figsize=(10, 7))
ax  = fig.add_subplot(111, projection="3d")

surf = ax.plot_surface(
    X_mesh, Y_mesh, intensity,
    cmap="viridis",
    edgecolor="none",
    alpha=0.8
)

# 축 레이블 & 제목
ax.set_xlabel("Time (s)")           # 이제 X축이 시간
ax.set_ylabel("Wavelength (nm)")    # Y축이 파장
ax.set_zlabel("Intensity")
ax.set_title("3D OES Intensity Surface\n(X: Time, Y: Wavelength)")

# 컬러바 추가
fig.colorbar(surf, ax=ax, pad=0.1, label="Intensity (a.u.)")

plt.tight_layout()
plt.show()
