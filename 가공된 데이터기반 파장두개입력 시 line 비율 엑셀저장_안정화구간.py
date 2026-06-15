"""
OES 라인비 계산 (안정화 구간만 평균)
- 안정화 시작: 어느 한 파장이라도 intensity >= THR
- 안정화 종료: 시작 이후 모든 파장이 intensity <= THR (그 '직전'까지 포함)
- 파장 매칭: 부동소수 허용 np.isclose(atol=0.01)
"""




import os
import numpy as np
import pandas as pd

file_sets = {
    "test1": {
        "1Torque": r"C:\Users\gusrb\OneDrive\Desktop\WGS 실험 DATA\20250808 분석 data\Ar 110 Source 1200 integration time10 total\Ar 110 Source 1200 integration time10 01_1.csv",
        "5Torque": r"C:\Users\gusrb\OneDrive\Desktop\WGS 실험 DATA\20250808 분석 data\Ar 110 Source 1200 integration time10 total\Ar 110 Source 1200 integration time10 05_1.csv",
        "10Torque": r"C:\Users\gusrb\OneDrive\Desktop\WGS 실험 DATA\20250808 분석 data\Ar 110 Source 1200 integration time10 total\Ar 110 Source 1200 integration time10 10_1.csv",
        "15Torque": r"C:\Users\gusrb\OneDrive\Desktop\WGS 실험 DATA\20250808 분석 data\Ar 110 Source 1200 integration time10 total\Ar 110 Source 1200 integration time10 15_1.csv",
    },
    "test2": {
        "1Torque": r"C:\Users\gusrb\OneDrive\Desktop\WGS 실험 DATA\20250808 분석 data\Ar 110 Source 1200 integration time10 total\Ar 110 Source 1200 integration time10 01_2.csv",
        "5Torque": r"C:\Users\gusrb\OneDrive\Desktop\WGS 실험 DATA\20250808 분석 data\Ar 110 Source 1200 integration time10 total\Ar 110 Source 1200 integration time10 05_2.csv",
        "10Torque": r"C:\Users\gusrb\OneDrive\Desktop\WGS 실험 DATA\20250808 분석 data\Ar 110 Source 1200 integration time10 total\Ar 110 Source 1200 integration time10 10_2.csv",
        "15Torque": r"C:\Users\gusrb\OneDrive\Desktop\WGS 실험 DATA\20250808 분석 data\Ar 110 Source 1200 integration time10 total\Ar 110 Source 1200 integration time10 15_2.csv",
    },
    "test3": {
        "1Torque": r"C:\Users\gusrb\OneDrive\Desktop\WGS 실험 DATA\20250808 분석 data\Ar 110 Source 1200 integration time10 total\Ar 110 Source 1200 integration time10 01_3.csv",
        "5Torque": r"C:\Users\gusrb\OneDrive\Desktop\WGS 실험 DATA\20250808 분석 data\Ar 110 Source 1200 integration time10 total\Ar 110 Source 1200 integration time10 05_3.csv",
        "10Torque": r"C:\Users\gusrb\OneDrive\Desktop\WGS 실험 DATA\20250808 분석 data\Ar 110 Source 1200 integration time10 total\Ar 110 Source 1200 integration time10 10_3.csv",
        "15Torque": r"C:\Users\gusrb\OneDrive\Desktop\WGS 실험 DATA\20250808 분석 data\Ar 110 Source 1200 integration time10 total\Ar 110 Source 1200 integration time10 15_3.csv",
    },
    "test4": {
        "1Torque": r"C:\Users\gusrb\OneDrive\Desktop\WGS 실험 DATA\20250808 분석 data\Ar 110 Source 1200 integration time10 total\Ar 110 Source 1200 integration time10 01_4.csv",
        "5Torque": r"C:\Users\gusrb\OneDrive\Desktop\WGS 실험 DATA\20250808 분석 data\Ar 110 Source 1200 integration time10 total\Ar 110 Source 1200 integration time10 05_4.csv",
        "10Torque": r"C:\Users\gusrb\OneDrive\Desktop\WGS 실험 DATA\20250808 분석 data\Ar 110 Source 1200 integration time10 total\Ar 110 Source 1200 integration time10 10_4.csv",
        "15Torque": r"C:\Users\gusrb\OneDrive\Desktop\WGS 실험 DATA\20250808 분석 data\Ar 110 Source 1200 integration time10 total\Ar 110 Source 1200 integration time10 15_4.csv",
    },
}

# 비교할 라인 (원하는 값으로 바꾸세요)
numerator_wavelength   = 427.68   # 분자
denominator_wavelength = 603.10   # 분모

# 안정화/매칭 설정
THR      = 100.0     # 안정화 임계값
WL_ATOL  = 1e-2      # 파장 매칭 허용오차(±0.01 nm)
N_CHANNELS = 8       # 채널 수 기대값

# 결과 저장 폴더 (없으면 생성)
OUT_DIR = r"C:\Users\gusrb\OneDrive\Desktop\WGS 실험 DATA\20250808 분석 data"
os.makedirs(OUT_DIR, exist_ok=True)

# ===================== 2) 안정화 구간 탐지 함수 =====================
def find_stable_slice(data_2d: pd.DataFrame, thr: float = THR):
    """
    data_2d: (time x wavelength) 데이터프레임 (모든 파장 포함)
    시작: 어떤 파장이라도 >= thr
    종료: 시작 이후 모든 파장이 <= thr 가 되는 첫 행의 '직전'까지 포함
    반환: slice(start, end+1). 시작을 못 찾으면 None
    """
    arr = data_2d.to_numpy(copy=False)
    if arr.size == 0:
        return None

    any_ge = (arr >= thr).any(axis=1)   # 행별: 하나라도 임계 이상
    all_le = (arr <= thr).all(axis=1)   # 행별: 모두 임계 이하

    starts = np.where(any_ge)[0]
    if starts.size == 0:
        return None  # 시작 조건 없음

    start = int(starts[0])

    after = np.arange(arr.shape[0]) > start
    ends_flag = np.where(all_le & after)[0]
    if ends_flag.size == 0:
        end = arr.shape[0] - 1  # 끝까지
    else:
        end = int(ends_flag[0] - 1)
        if end < start:
            end = start

    return slice(start, end)

# ===================== 3) 그룹 처리 함수 =====================
def process_file_group(group_name, file_dict, num_wl, den_wl):
    result_df = pd.DataFrame()

    for torque_label, file_path in file_dict.items():
        # 1) 읽기 (행0: 파장, 행1~: 데이터)
        df = pd.read_csv(file_path, header=None)
        wavelengths = pd.to_numeric(df.iloc[0, 1:], errors='coerce').values
        data = df.iloc[1:, 1:].apply(pd.to_numeric, errors='coerce').fillna(0.0).reset_index(drop=True)

        # 2) 안정화 구간만 선택
        sl = find_stable_slice(data, thr=THR)
        if sl is None or (sl.stop - sl.start) <= 0:
            print(f"⚠ [{group_name} / {torque_label}] 안정화 구간 미탐지 → 전체 구간 사용")
            data_use = data
        else:
            data_use = data.iloc[sl]
            print(f"✅ [{group_name} / {torque_label}] 안정화 rows: {sl.start}~{sl.stop-1} (총 {sl.stop-sl.start}개)")

        # 3) 파장 인덱스(각 채널) 찾기: np.isclose로 안전 매칭
        num_idx_all = np.where(np.isclose(wavelengths, num_wl, atol=WL_ATOL))[0]
        den_idx_all = np.where(np.isclose(wavelengths, den_wl, atol=WL_ATOL))[0]
        if len(num_idx_all) == 0 or len(den_idx_all) == 0:
            raise ValueError(f"[{group_name}/{torque_label}] 지정 파장 미발견: "
                             f"num={num_wl}, den={den_wl}")

        ch_count = min(N_CHANNELS, len(num_idx_all), len(den_idx_all))
        if ch_count < N_CHANNELS:
            print(f"ℹ [{group_name}/{torque_label}] 매칭 채널 {ch_count}개(기대 {N_CHANNELS}) → 나머지는 NaN")

        # 4) 채널별 평균/비율/로그 계산 (안정화 구간만)
        out = {"Log": [], "Ratio": [], str(num_wl): [], str(den_wl): []}
        for ch in range(N_CHANNELS):
            if ch < ch_count:
                num_col = int(num_idx_all[ch])
                den_col = int(den_idx_all[ch])

                num_avg = float(data_use.iloc[:, num_col].mean())
                den_avg = float(data_use.iloc[:, den_col].mean())

                ratio   = np.nan if den_avg == 0 else (num_avg / den_avg)
                log_val = np.nan if (ratio is None or not np.isfinite(ratio) or ratio <= 0) else np.log10(ratio)

                out["Log"].append(log_val)
                out["Ratio"].append(ratio)
                out[str(num_wl)].append(num_avg)
                out[str(den_wl)].append(den_avg)
            else:
                out["Log"].append(np.nan)
                out["Ratio"].append(np.nan)
                out[str(num_wl)].append(np.nan)
                out[str(den_wl)].append(np.nan)

        section_df = pd.DataFrame(out, index=[f"channel{i}" for i in range(1, 9)]).T.reset_index()
        section_df = section_df.rename(columns={"index": "Channel"})
        header_row = pd.DataFrame([[torque_label] + [""] * (section_df.shape[1] - 1)], columns=section_df.columns)

        result_df = pd.concat([result_df, header_row, section_df], ignore_index=True)

    return result_df

# ===================== 4) 실행 & 저장 =====================
final_df = pd.DataFrame()
for group_name, file_group in file_sets.items():
    df = process_file_group(group_name, file_group, numerator_wavelength, denominator_wavelength)
    header = pd.DataFrame([[group_name] + [""] * (df.shape[1] - 1)], columns=df.columns)
    final_df = pd.concat([final_df, header, df], ignore_index=True)

output_path = os.path.join(
    OUT_DIR,
    f"{numerator_wavelength}_{denominator_wavelength}OES_Ratio_Summary_Vertical.xlsx"
)
with pd.ExcelWriter(output_path, engine='xlsxwriter') as writer:
    final_df.to_excel(writer, index=False, sheet_name="Summary")

print(f"✅ 엑셀 저장 완료: {output_path}")