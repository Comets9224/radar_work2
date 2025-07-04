# cfar_test.py
import numpy as np
import matplotlib.pyplot as plt
import os # <--- 添加这一行来导入 os 模块
import sys # 用于修改模块搜索路径，以便导入config
import pathlib

# 假设cfar_test.py在modules文件夹内，config.py在上一级目录
# 如果cfar_test.py在项目根目录，则不需要修改sys.path
current_dir = pathlib.Path(__file__).parent
project_root = current_dir.parent
sys.path.append(str(project_root))

import config as cfg # 现在可以正确导入config了


def generate_1d_power_signal(length, noise_mean_power, target_pos, snr_db):
    noise_power = np.random.exponential(scale=noise_mean_power, size=length)
    signal_power = noise_power.copy()
    if target_pos is not None and 0 <= target_pos < length:
        target_signal_power_val = noise_mean_power * (10**(snr_db / 10))
        signal_power[target_pos] += target_signal_power_val
    return signal_power, noise_power

def ca_cfar_detector(signal_power, num_guard_cells, num_ref_cells, pfa_target):
    detections = []
    threshold_values = []
    signal_length = len(signal_power)
    N_ref_total = 2 * num_ref_cells

    if N_ref_total == 0:
        alpha = 10
    else:
        alpha = N_ref_total * (pfa_target**(-1/N_ref_total) - 1)

    for i in range(num_ref_cells + num_guard_cells, signal_length - num_ref_cells - num_guard_cells):
        lagging_window_start = i - num_guard_cells - num_ref_cells
        lagging_window_end = i - num_guard_cells -1
        lagging_window = signal_power[lagging_window_start : lagging_window_end + 1]

        leading_window_start = i + num_guard_cells + 1
        leading_window_end = i + num_guard_cells + num_ref_cells
        leading_window = signal_power[leading_window_start : leading_window_end + 1]

        sum_lagging = np.sum(lagging_window)
        sum_leading = np.sum(leading_window)
        noise_power_estimate = (sum_lagging + sum_leading) / N_ref_total

        threshold = alpha * noise_power_estimate
        threshold_values.append(threshold)

        if signal_power[i] > threshold:
            detections.append(i)
    return detections, threshold_values

if __name__ == '__main__':
    N_CELLS = 200
    NOISE_MEAN_POWER = 1
    TARGET_POS = 100
    SNR_DB = 15
    GUARD_CELLS_ONE_SIDE = 2
    REF_CELLS_ONE_SIDE = 10
    PFA_TARGET = 1e-3

    print(f"--- CFAR Test Parameters ---")
    print(f"N_CELLS: {N_CELLS}, NOISE_MEAN_POWER: {NOISE_MEAN_POWER}")
    print(f"TARGET_POS: {TARGET_POS}, SNR_DB: {SNR_DB}")
    print(f"GUARD_CELLS_ONE_SIDE: {GUARD_CELLS_ONE_SIDE}, REF_CELLS_ONE_SIDE: {REF_CELLS_ONE_SIDE}")
    print(f"PFA_TARGET: {PFA_TARGET}")

    signal_with_target, noise_only_signal = generate_1d_power_signal(
        N_CELLS, NOISE_MEAN_POWER, TARGET_POS, SNR_DB
    )
    detections, cfar_thresholds = ca_cfar_detector(
        signal_with_target, GUARD_CELLS_ONE_SIDE, REF_CELLS_ONE_SIDE, PFA_TARGET
    )

    plt.figure(figsize=(12, 6))
    plt.plot(signal_with_target, label='Signal + Noise Power', color='blue', linewidth=1)
    threshold_x_axis = range(REF_CELLS_ONE_SIDE + GUARD_CELLS_ONE_SIDE,
                             N_CELLS - REF_CELLS_ONE_SIDE - GUARD_CELLS_ONE_SIDE)
    plt.plot(list(threshold_x_axis), cfar_thresholds, 'r--', label='CA-CFAR Threshold', linewidth=1.5)

    if TARGET_POS is not None:
         plt.axvline(TARGET_POS, color='green', linestyle=':', label=f'True Target at {TARGET_POS}', linewidth=1.5)

    if detections:
        plt.scatter(detections, [signal_with_target[i] for i in detections],
                    c='red', marker='x', s=100, label='Detections', zorder=5)

    plt.xlabel('Range Cell Index')
    plt.ylabel('Power')
    plt.title(f'1D CA-CFAR Detection (SNR={SNR_DB}dB, PFA={PFA_TARGET})')
    plt.legend()
    plt.grid(True)
    plt.ylim(bottom=0)

    # 使用 config.py 中定义的路径来保存图像
    os.makedirs(cfg.OUTPUT_PLOTS_DIR, exist_ok=True)
    cfar_plot_path = os.path.join(cfg.OUTPUT_PLOTS_DIR, "cfar_detection_demo.png")
    plt.savefig(cfar_plot_path)
    print(f"\nCFAR detection plot saved to {cfar_plot_path}")
    plt.show()

    print(f"Detections at cell indices: {detections}")
    if TARGET_POS is not None and TARGET_POS in detections: # 确保TARGET_POS不是None
        print(f"Target at cell {TARGET_POS} was correctly detected.")
    elif TARGET_POS is not None:
        print(f"Target at cell {TARGET_POS} was missed.")
    else:
        print("No target was set in this CFAR test run.")


    false_alarms_on_noise, _ = ca_cfar_detector(
        noise_only_signal, GUARD_CELLS_ONE_SIDE, REF_CELLS_ONE_SIDE, PFA_TARGET
    )
    num_testable_cells = N_CELLS - 2 * (REF_CELLS_ONE_SIDE + GUARD_CELLS_ONE_SIDE)
    if num_testable_cells > 0:
        actual_pfa = len(false_alarms_on_noise) / num_testable_cells
        print(f"Actual PFA on noise-only signal: {actual_pfa:.2e} (target PFA: {PFA_TARGET:.1e})")
    else:
        print("Not enough cells to test PFA reliably with current window settings.")
