import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def load_channelwise_avg_precise(filepath: str):
    """
    파일 전체 구간을 평균해서 채널별 intensity 계산 (안정화 구간 탐색 제거)
    """
    df = pd.read_csv(filepath, header=None)

    # 전체 파장 (첫 번째 행) + intensity 데이터 (두 번째 행 이후)
    all_wavelengths = df.iloc[0, 1:].astype(float).values
    intensity_data = df.iloc[1:, 1:].astype(float).values

    # ✅ 전체 구간 평균 (안정화 구간 탐색 없이)
    avg_intensity = intensity_data.mean(axis=0)

    # 채널 개수
    num_channels = 8
    channel_data = {ch: [] for ch in range(num_channels)}

    for ch in range(num_channels):
        ch_wls = all_wavelengths[ch::num_channels]      # 해당 채널 파장
        ch_vals = avg_intensity[ch::num_channels]       # 해당 채널 intensity
        df_ch = pd.DataFrame({'Wavelength': ch_wls, 'Intensity': ch_vals})
        grouped = df_ch.groupby('Wavelength').mean().sort_index()
        channel_data[ch] = grouped

    # 공통 파장 기준 재정렬
    all_unique_wavelengths = sorted(set().union(*[df.index for df in channel_data.values()]))
    intensity_matrix = np.zeros((len(all_unique_wavelengths), num_channels))

    for ch in range(num_channels):
        interp = channel_data[ch]
        for i, wl in enumerate(all_unique_wavelengths):
            if wl in interp.index:
                intensity_matrix[i, ch] = interp.loc[wl, 'Intensity']

    return np.array(all_unique_wavelengths), intensity_matrix


def plot_comparative_channelwise_spectra_precise(file_dict):
    """
    여러 Torque 조건 파일을 받아 채널별 비교 plot 생성 (총 8개)
    """
    num_channels = 8
    fig, axes = plt.subplots(4, 2, figsize=(14, 10), sharex=True)
    axes = axes.flatten()

    for label, filepath in file_dict.items():
        try:
            wl, intensity_matrix = load_channelwise_avg_precise(filepath)
            for ch in range(num_channels):
                axes[ch].plot(wl, intensity_matrix[:, ch], label=label)
        except Exception as e:
            print(f"❌ {label} 오류 발생: {e}")

    for ch in range(num_channels):
        axes[ch].set_title(f"Channel {ch+1}")
        axes[ch].set_ylabel("Intensity (a.u.)")
        axes[ch].grid(True)
        axes[ch].legend()

    axes[-1].set_xlabel("Wavelength (nm)")
    plt.suptitle("Channel-wise Spectra Comparison (Full Range)", fontsize=14)
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.show()

# ✅ 사용 예시
file_dict = {
    "1 Torque": r"C:\Users\gusrb\Desktop\WGS 실험 DATA\20250725 분석data\Ar Source 1torque test2.csv",
    "5 Torque": r"C:\Users\gusrb\Desktop\WGS 실험 DATA\20250725 분석data\Ar Source 5torque test2.csv",
    "10 Torque": r"C:\Users\gusrb\Desktop\WGS 실험 DATA\20250725 분석data\Ar Source 10torque test2.csv",
    "15 Torque": r"C:\Users\gusrb\Desktop\WGS 실험 DATA\20250725 분석data\Ar Source 15torque test2.csv"
}

plot_comparative_channelwise_spectra_precise(file_dict)