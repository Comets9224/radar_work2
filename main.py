# main.py
import numpy as np
import os
import matplotlib.pyplot as plt
import config as cfg
from modules import data_generator, detector, visualizer
from modules.tracker import Tracker


def run_simulation():
    """主仿真函数，调度所有模块"""
    # 打印配置和模型信息
    # cfg.print_config_summary() # 如果您在config.py中实现了这个函数
    data_generator.print_motion_equations()
    print("\nStarting Radar Target Detection and Tracking System Simulation...")

    # --- 准备输出目录 ---
    os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)
    os.makedirs(cfg.OUTPUT_DATA_DIR, exist_ok=True)
    os.makedirs(cfg.OUTPUT_PLOTS_DIR, exist_ok=True)

    # --- 阶段二：生成仿真数据 ---
    print("\n--- Phase 2: Generating Simulation Data ---")
    true_trajs, obs_traj, measurements_per_step = data_generator.generate_full_dataset(cfg)
    print(f"Generated {len(measurements_per_step)} frames of data.")

    # --- 阶段三 & 四：检测与跟踪 ---
    print("\n--- Phase 3 & 4: Detection and Tracking ---")
    tracker = Tracker()
    all_detections_over_time = []
    all_tracks_over_time = []

    for k in range(len(measurements_per_step)):
        current_time = k * cfg.DT
        observer_state_k = obs_traj[k]
        raw_measurements_polar_k = measurements_per_step[k]

        # 阶段三: 检测
        detected_clusters_k = detector.detect_targets_by_clustering(
            raw_measurements_polar_k,
            observer_state_k,
            cfg
        )
        all_detections_over_time.append(detected_clusters_k)

        # 阶段四: 跟踪
        tracked_info_k = tracker.step(
            detected_clusters_k,
            observer_state_k
        )
        all_tracks_over_time.append(tracked_info_k)

        # 每隔20帧打印一次进度
        if (k + 1) % 20 == 0 or k == len(measurements_per_step) - 1:
            num_raw = len(raw_measurements_polar_k)
            num_clusters = len(detected_clusters_k)
            num_active_tracks = len(tracker.tracks)
            num_confirmed = sum(1 for t in tracker.tracks if t.state == 'Confirmed')
            print(
                f"  Time {current_time:.1f}s: RawMeas={num_raw}, Detections={num_clusters}, ActiveTracks={num_active_tracks} (Confirmed={num_confirmed})")

    # --- 阶段五：可视化 ---
    print("\n--- Phase 5: Visualization ---")

    # 从最后一帧的跟踪结果中提取每个航迹的完整历史用于绘图
    final_tracks_for_plot = []
    if all_tracks_over_time and all_tracks_over_time[-1]:
        final_tracks_for_plot = all_tracks_over_time[-1]

    visualizer.plot_results(
        observer_trajectory=obs_traj,
        true_target_trajectories=true_trajs,
        measurements_over_time=measurements_per_step,
        detections_over_time=all_detections_over_time,
        tracks_over_time=final_tracks_for_plot
    )

    print("\nSimulation complete. Plot saved in 'output/plots/' directory.")
    # plt.show() # 如果希望在运行时立即显示图像，可以取消此行注释


if __name__ == '__main__':
    run_simulation()