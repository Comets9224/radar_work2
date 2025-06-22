# main.py
import numpy as np
import os
import matplotlib.pyplot as plt
import config as cfg
from modules import data_generator, detector, visualizer
from modules.tracker import Tracker
from modules.track import Track
from scipy.optimize import linear_sum_assignment
from collections import defaultdict

roc_points = []  # List of (pfa, pd)


def calculate_num_resolution_cells(config_obj):
    """计算雷达视场内的分辨单元总数 (近似)"""
    if config_obj.RADAR_RANGE_RESOLUTION_M <= 0 or config_obj.RADAR_AZIMUTH_RESOLUTION_DEG <= 0:
        print("Warning: Radar resolution not properly defined for P_fa calculation. Returning large number.")
        return 1e6  # 返回一个较大的数避免除零

    num_range_bins = (config_obj.RADAR_MAX_RANGE - config_obj.RADAR_MIN_RANGE) / config_obj.RADAR_RANGE_RESOLUTION_M

    # 对于扇形区域，方位单元数可以近似为 FOV / 分辨率
    # 更精确的计算会考虑不同距离上的弧长，但这里用简化版
    num_azimuth_bins = config_obj.RADAR_FOV_DEG / config_obj.RADAR_AZIMUTH_RESOLUTION_DEG

    # 确保至少有一个单元
    num_range_bins = max(1, np.floor(num_range_bins))
    num_azimuth_bins = max(1, np.floor(num_azimuth_bins))

    # print(f"Debug: Num Range Bins: {num_range_bins}, Num Azimuth Bins: {num_azimuth_bins}")
    return num_range_bins * num_azimuth_bins


def run_single_simulation_for_roc(current_eps_value, true_target_trajs_global, obs_traj_global,
                                  measurements_per_step_polar_global, num_resolution_cells_per_frame):
    """
    为单个EPS值运行检测和初步评估，返回 (pfa, pd)。
    """
    print(f"\n--- Running simulation for DBSCAN_EPS = {current_eps_value} ---")
    cfg.DBSCAN_EPS = current_eps_value

    num_simulation_steps = len(measurements_per_step_polar_global)
    true_target_lifetimes = [len(traj) for traj in true_target_trajs_global]
    total_true_target_instances_in_sim = sum(true_target_lifetimes)  # 真实目标出现的总次数（每帧算一次）

    total_dbscan_fp_clusters = 0
    total_dbscan_tp_clusters = 0

    for k in range(num_simulation_steps):
        observer_state_k = obs_traj_global[k]
        raw_measurements_polar_k_current_step = measurements_per_step_polar_global[k]

        detected_clusters_info_k = detector.detect_targets_by_clustering(
            raw_measurements_polar_k_current_step,
            observer_state_k,
            cfg
        )

        true_target_states_at_k_for_eval = []
        num_true_targets_in_frame_k = 0  # 当前帧实际存在的真实目标数量
        for target_idx, true_traj_single_target in enumerate(true_target_trajs_global):
            if k < len(true_traj_single_target):
                # 检查目标是否在雷达基本覆盖范围内（更严格的P_d应考虑此）
                # 这里简化，假设所有存在的真实目标都是P_d计算的分母的一部分
                num_true_targets_in_frame_k += 1
                true_target_states_at_k_for_eval.append({
                    "id": target_idx,
                    "pos_xy": true_traj_single_target[k, :2]
                })

        detected_cluster_associated_to_true_count_k = 0
        if detected_clusters_info_k and true_target_states_at_k_for_eval:
            num_det_clusters = len(detected_clusters_info_k)
            cost_matrix_dbscan_true = np.full((num_det_clusters, len(true_target_states_at_k_for_eval)), np.inf)
            for i, det_cluster in enumerate(detected_clusters_info_k):
                det_pos = np.array(det_cluster['position_global'])
                for j, true_target_data in enumerate(true_target_states_at_k_for_eval):
                    true_pos = true_target_data["pos_xy"]
                    dist = np.linalg.norm(det_pos - true_pos)
                    cost_matrix_dbscan_true[i, j] = dist

            det_indices, true_indices = linear_sum_assignment(cost_matrix_dbscan_true)
            # 关联门限，可以考虑与EPS联动或固定一个较宽松的值
            association_threshold_roc = current_eps_value * 1.5  # 示例
            # association_threshold_roc = 5.0 # 或者一个固定值

            # 确保每个真实目标最多只被一个检测簇关联 (匈牙利算法已保证检测簇最多关联一个真值)
            # 我们需要统计的是TP (检测簇关联到真值) 和 FN (真值未被任何检测簇关联)

            # TP: 检测簇关联到真值
            # FP: 检测簇未关联到真值
            # FN: 真值未被检测簇关联

            associated_true_target_indices_k = set()
            for d_idx, t_idx in zip(det_indices, true_indices):
                if cost_matrix_dbscan_true[d_idx, t_idx] < association_threshold_roc:
                    detected_cluster_associated_to_true_count_k += 1
                    # 记录下这个真实目标被关联了 (它的ID是 true_target_states_at_k_for_eval[t_idx]['id'])
                    associated_true_target_indices_k.add(true_target_states_at_k_for_eval[t_idx]['id'])

        total_dbscan_tp_clusters += detected_cluster_associated_to_true_count_k
        total_dbscan_fp_clusters += (len(detected_clusters_info_k) - detected_cluster_associated_to_true_count_k)

        # FN的计算: 当前帧存在的真实目标数 - 当前帧被关联上的真实目标数
        # total_fn += (num_true_targets_in_frame_k - len(associated_true_target_indices_k))
        # P_d 的分母是总的真实目标出现次数，TP是检测簇关联到真值的总次数

    # --- 计算 P_d ---
    # P_d = TP / (TP + FN)
    # TP = total_dbscan_tp_clusters
    # (TP + FN) = total_true_target_instances_in_sim (所有帧中真实目标出现的总次数)
    overall_pd = total_dbscan_tp_clusters / total_true_target_instances_in_sim if total_true_target_instances_in_sim > 0 else 0

    # --- 计算 P_fa ---
    # P_fa = FP / (N_fa_opp_per_frame * num_simulation_steps)
    # FP = total_dbscan_fp_clusters
    total_false_alarm_opportunities = num_resolution_cells_per_frame * num_simulation_steps
    overall_pfa = total_dbscan_fp_clusters / total_false_alarm_opportunities if total_false_alarm_opportunities > 0 else 0

    print(f"  For EPS={current_eps_value:.2f}: P_fa={overall_pfa:.3e}, P_d={overall_pd:.3%}")  # P_fa用科学计数法
    return overall_pfa, overall_pd


def run_simulation_with_roc_generation():
    data_generator.print_motion_equations()
    print("\nStarting Radar Target Detection and Tracking System Simulation with ROC Generation...")

    os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)
    os.makedirs(cfg.OUTPUT_DATA_DIR, exist_ok=True)
    os.makedirs(cfg.OUTPUT_PLOTS_DIR, exist_ok=True)

    print("\n--- Phase 2: Generating Simulation Data (once) ---")
    true_target_trajs_g, obs_traj_g, measurements_per_step_polar_g = data_generator.generate_full_dataset(cfg)
    print(f"Generated {len(measurements_per_step_polar_g)} frames of data.")

    # 计算一次分辨单元数
    num_res_cells = calculate_num_resolution_cells(cfg)
    print(f"Estimated number of resolution cells per frame: {num_res_cells}")
    if num_res_cells <= 0:
        print("ERROR: Number of resolution cells is zero or negative. P_fa calculation will be incorrect.")
        return

    eps_values_for_roc = [1.0, 1.5, 2.0, 2.7, 3.5, 4.5, 6.0, 8.0, 10.0, 12.0, 15.0]  # 调整EPS范围
    original_dbscan_eps = cfg.DBSCAN_EPS

    global roc_points
    roc_points = []

    for eps_val in eps_values_for_roc:
        # 注意：这里传入了 num_res_cells
        pfa_val, pd_val = run_single_simulation_for_roc(eps_val, true_target_trajs_g, obs_traj_g,
                                                        measurements_per_step_polar_g, num_res_cells)
        roc_points.append((pfa_val, pd_val))

    roc_points.sort(key=lambda x: x[0])  # 按P_fa排序

    print(f"\n--- Running final full simulation with original DBSCAN_EPS = {original_dbscan_eps} ---")
    cfg.DBSCAN_EPS = original_dbscan_eps

    # ... (省略原来main函数中完整的跟踪和评估逻辑，你需要将其复制回来) ...
    # ... 确保这部分使用恢复后的 cfg.DBSCAN_EPS ...
    # --- 示例：假设我们已经运行了完整的仿真并得到了这些结果 ---
    Track._next_id = 1
    tracker_final = Tracker()
    all_detections_info_over_time_final = []
    all_tracks_info_over_time_final = []
    time_vector_final = []
    position_rmse_over_time_final = []
    velocity_rmse_over_time_final = []

    for k_final in range(len(measurements_per_step_polar_g)):
        observer_state_k_final = obs_traj_g[k_final]
        raw_measurements_polar_k_current_step_final = measurements_per_step_polar_g[k_final]
        detected_clusters_info_k_final = detector.detect_targets_by_clustering(
            raw_measurements_polar_k_current_step_final, observer_state_k_final, cfg)
        all_detections_info_over_time_final.append(detected_clusters_info_k_final)
        tracked_info_for_frame_k_final = tracker_final.step(
            detected_clusters_info_k_final, observer_state_k_final)
        all_tracks_info_over_time_final.append(tracked_info_for_frame_k_final)
        time_vector_final.append(k_final * cfg.DT)
        position_rmse_over_time_final.append(np.nan)
        velocity_rmse_over_time_final.append(np.nan)

    detections_for_plot_global_xy_final = []
    for frame_detections_info in all_detections_info_over_time_final:
        current_frame_xy_detections = []
        for det_info in frame_detections_info:
            current_frame_xy_detections.append(det_info['position_global'])
        detections_for_plot_global_xy_final.append(current_frame_xy_detections)
    # --- 完整仿真运行结束 ---

    print("\n--- Phase 5: Visualization ---")
    visualizer.plot_results_with_roc(
        observer_trajectory=obs_traj_g,
        true_target_trajectories=true_target_trajs_g,
        measurements_over_time_polar=measurements_per_step_polar_g,
        detections_over_time_global_xy_list_of_lists=detections_for_plot_global_xy_final,
        all_tracks_info_over_time=all_tracks_info_over_time_final,
        time_vector=time_vector_final,
        position_rmse_over_time=position_rmse_over_time_final,
        velocity_rmse_over_time=velocity_rmse_over_time_final,
        roc_curve_points=roc_points  # roc_points 现在是 (pfa, pd)
    )
    print("\nSimulation complete. Plots saved in 'output/plots/' directory.")


if __name__ == '__main__':
    run_simulation_with_roc_generation()