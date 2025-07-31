import os
import glob
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt

# 1️⃣ CSV 파일 자동 읽기
DATA_DIR = r"C:\Users\gusrb\Desktop\WGS 실험 DATA\20250725 분석data\VI probe data" #폴더경로 입력
csv_files = sorted(glob.glob(os.path.join(DATA_DIR, "*.csv"), recursive=False))
print(f"✅ 읽은 파일: {csv_files}")

if not csv_files:
    raise FileNotFoundError("CSV 파일이 없습니다!")

# 2️⃣ CSV 읽어서 dict 저장
data = {}
min_length = None   # 가장 짧은 CSV 길이 저장
for fp in csv_files:
    label = Path(fp).stem
    df = pd.read_csv(fp, sep=None, engine="python")
    data[label] = df
    # CSV 길이가 다를 때 최소 길이에 맞춰 자르기 위해 확인
    if min_length is None or len(df) < min_length:
        min_length = len(df)

print(f"📏 모든 CSV를 {min_length} row 기준으로 맞춥니다.")

# 3️⃣ 첫 번째 파일 기준으로 Time 컬럼과 파라미터 추출
sample_df = next(iter(data.values()))
time_col = "Time" if "Time" in sample_df.columns else sample_df.columns[0]
param_cols = [c for c in sample_df.columns if c != time_col]

# 4️⃣ 색상 팔레트
colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]

# 5️⃣ 파라미터별 그래프 생성
for col in param_cols:
    plt.figure(figsize=(10,4))

    for i, (label, df) in enumerate(data.items()):
        # ✅ 공통 인덱스 길이에 맞춰 잘라서 plot
        plt.plot(range(min_length), df[col].iloc[:min_length],
                 label=label,
                 color=colors[i % len(colors)],
                 linewidth=1.5)

    plt.title(col, fontsize=14)
    plt.xlabel("Sample Index")   # ✅ X축은 Sample Index
    plt.ylabel(col)
    plt.legend(title="Condition")
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.show()