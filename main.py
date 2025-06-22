# main.py
import numpy as np
import os
import matplotlib.pyplot as plt
import config as cfg
from modules import data_generator, detector, visualizer
from modules.tracker import Tracker
from modules.track import Track  # Import Track to reset _next_id if needed
from scipy.optimize import linear_sum_assignment  # <--- 新增导入


def run_simulation():
    """主仿真函数，调度所有模块"""
    # cfg.print_config_summary() # 取消注释以打印配置
    data_generator.print_motion_equations()
    print("\nStarting Radar Target Detection and Tracking System Simulation...")

    os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)
    os.makedirs(cfg.OUTPUT_DATA_DIR, exist_ok=True)
    os.makedirs(cfg.OUTPUT_PLOTS_DIR, exist_ok=True)

    print("\n--- Phase 2: Generating Simulation Data ---")
    true_target_trajs, obs_traj, measurements_per_step_polar = data_generator.generate_full_dataset(cfg)
    print(f"Generated {len(measurements_per_step_polar)} frames of data.")

    print("\n--- Phase 3 & 4: Detection and Tracking ---")
    Track._next_id = 1  # 重置航迹ID计数器，以便每次仿真从1开始
    tracker = Tracker()
    all_detections_info_over_time = []
    all_tracks_info_over_time = []

    # 用于存储RMSE计算所需的数据
    all_time_steps_data_for_rmse = []
    RMSE_ASSOCIATION_GATE = cfg.GATING_THRESHOLD_EUCLIDEAN * 2.5  # RMSE关联门限，可调整

    num_simulation_steps = len(measurements_per_step_polar)

    for k in range(num_simulation_steps):
        current_time = k * cfg.DT
        observer_state_k = obs_traj[k]
        raw_measurements_polar_k_current_step = measurements_per_step_polar[k]

        detected_clusters_info_k = detector.detect_targets_by_clustering(
            raw_measurements_polar_k_current_step,
            observer_state_k,
            cfg
        )
        all_detections_info_over_time.append(detected_clusters_info_k)

        tracked_info_for_frame_k = tracker.step(
            detected_clusters_info_k,
            observer_state_k
        )
        all_tracks_info_over_time.append(tracked_info_for_frame_k)

        # --- RMSE 数据收集 ---
        true_target_states_at_k_for_rmse = []
        for target_idx, true_traj_single_target in enumerate(true_target_trajs):
            if k < len(true_traj_single_target):
                # 真实目标状态，取前4个元素 (px, py, vx, vy)
                true_target_states_at_k_for_rmse.append({
                    "id": target_idx,  # 使用索引作为临时ID
                    "state_xy_vxvy": true_traj_single_target[k, :4]
                })

        confirmed_tracks_at_k_for_rmse = []
        for track_info in tracked_info_for_frame_k:
            if track_info['state'] == 'Confirmed' and track_info['history']:
                # 确认航迹的最新状态 (px, py, vx, vy)
                confirmed_tracks_at_k_for_rmse.append({
                    "id": track_info['id'],
                    "state_xy_vxvy": np.array(track_info['history'][-1][:4])
                })

        step_pos_sq_errors = []
        step_vel_sq_errors = []

        if confirmed_tracks_at_k_for_rmse and true_target_states_at_k_for_rmse:
            num_confirmed_tracks = len(confirmed_tracks_at_k_for_rmse)
            num_true_targets_curr = len(true_target_states_at_k_for_rmse)

            cost_matrix_rmse = np.full((num_confirmed_tracks, num_true_targets_curr), np.inf)

            for i, confirmed_track_data in enumerate(confirmed_tracks_at_k_for_rmse):
                est_pos = confirmed_track_data["state_xy_vxvy"][:2]  # 估计位置 (px, py)
                for j, true_target_data in enumerate(true_target_states_at_k_for_rmse):
                    true_pos = true_target_data["state_xy_vxvy"][:2]  # 真实位置 (px, py)
                    dist = np.linalg.norm(est_pos - true_pos)
                    cost_matrix_rmse[i, j] = dist

            # 使用匈牙利算法进行关联 (已确认航迹 <-> 真实目标)
            track_indices_rmse, true_target_indices_rmse = linear_sum_assignment(cost_matrix_rmse)

            for tr_idx, gt_idx in zip(track_indices_rmse, true_target_indices_rmse):
                if cost_matrix_rmse[tr_idx, gt_idx] < RMSE_ASSOCIATION_GATE:
                    est_state = confirmed_tracks_at_k_for_rmse[tr_idx]["state_xy_vxvy"]
                    true_state = true_target_states_at_k_for_rmse[gt_idx]["state_xy_vxvy"]

                    # 位置误差平方: (est_px - true_px)^2 + (est_py - true_py)^2
                    pos_error_sq = np.sum((est_state[:2] - true_state[:2]) ** 2)
                    # 速度误差平方: (est_vx - true_vx)^2 + (est_vy - true_vy)^2
                    vel_error_sq = np.sum((est_state[2:4] - true_state[2:4]) ** 2)

                    step_pos_sq_errors.append(pos_error_sq)
                    step_vel_sq_errors.append(vel_error_sq)

        all_time_steps_data_for_rmse.append({
            "time": current_time,
            "pos_sq_errors": step_pos_sq_errors,
            "vel_sq_errors": step_vel_sq_errors
        })
        # --- RMSE 数据收集结束 ---

        if (k + 1) % 20 == 0 or k == num_simulation_steps - 1:
            num_raw = len(raw_measurements_polar_k_current_step)
            num_detected_clusters = len(detected_clusters_info_k)
            num_active_tracks = len(tracker.tracks)
            num_confirmed = sum(
                1 for t_info in tracked_info_for_frame_k if t_info['state'] == 'Confirmed')
            print(
                f"  Time {current_time:.1f}s: RawMeas={num_raw}, Detections={num_detected_clusters}, ActiveTracks={num_active_tracks} (Confirmed={num_confirmed})")

    # --- 计算最终的RMSE值 ---
    time_vector = [data['time'] for data in all_time_steps_data_for_rmse]
    position_rmse_over_time = []
    velocity_rmse_over_time = []

    for data in all_time_steps_data_for_rmse:
        if data['pos_sq_errors']:  # 如果当前时刻有关联上的航迹和真值
            mean_pos_sq_error = np.mean(data['pos_sq_errors'])
            position_rmse_over_time.append(np.sqrt(mean_pos_sq_error))
        else:
            position_rmse_over_time.append(np.nan)  # 没有有效误差数据，记为NaN

        if data['vel_sq_errors']:
            mean_vel_sq_error = np.mean(data['vel_sq_errors'])
            velocity_rmse_over_time.append(np.sqrt(mean_vel_sq_error))
        else:
            velocity_rmse_over_time.append(np.nan)

    print("\n--- Phase 5: Visualization ---")

    detections_for_plot_global_xy = []
    for frame_detections_info in all_detections_info_over_time:
        current_frame_xy_detections = []
        for det_info in frame_detections_info:
            current_frame_xy_detections.append(det_info['position_global'])
        detections_for_plot_global_xy.append(current_frame_xy_detections)

    visualizer.plot_results(
        observer_trajectory=obs_traj,
        true_target_trajectories=true_target_trajs,
        measurements_over_time_polar=measurements_per_step_polar,
        detections_over_time_global_xy_list_of_lists=detections_for_plot_global_xy,
        all_tracks_info_over_time=all_tracks_info_over_time,
        # 新增传递给绘图函数的数据
        time_vector=time_vector,
        position_rmse_over_time=position_rmse_over_time,
        velocity_rmse_over_time=velocity_rmse_over_time
    )

    print("\nSimulation complete. Plot saved in 'output/plots/' directory.")
    # plt.show() # visualizer.plot_results 内部已经调用了 plt.show()


if __name__ == '__main__':
    run_simulation()