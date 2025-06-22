# main.py
import numpy as np
import os
import matplotlib.pyplot as plt
import config as cfg
from modules import data_generator, detector, visualizer
from modules.tracker import Tracker
from modules.track import Track
from scipy.optimize import linear_sum_assignment
from collections import defaultdict # <--- 新增导入

def run_simulation():
    """主仿真函数，调度所有模块"""
    data_generator.print_motion_equations()
    print("\nStarting Radar Target Detection and Tracking System Simulation...")

    os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)
    os.makedirs(cfg.OUTPUT_DATA_DIR, exist_ok=True)
    os.makedirs(cfg.OUTPUT_PLOTS_DIR, exist_ok=True)

    print("\n--- Phase 2: Generating Simulation Data ---")
    true_target_trajs, obs_traj, measurements_per_step_polar = data_generator.generate_full_dataset(cfg)
    print(f"Generated {len(measurements_per_step_polar)} frames of data.")
    num_simulation_steps = len(measurements_per_step_polar)
    num_true_targets = len(true_target_trajs)

    print("\n--- Phase 3 & 4: Detection and Tracking ---")
    Track._next_id = 1
    tracker = Tracker()
    all_detections_info_over_time = []
    all_tracks_info_over_time = []

    all_time_steps_data_for_rmse = []
    RMSE_ASSOCIATION_GATE = cfg.GATING_THRESHOLD_EUCLIDEAN * 2.5

    # --- 用于指标计算的辅助数据结构 ---
    # 记录每个真实目标在每一帧是否被“覆盖”（即有对应的已确认航迹关联上）
    # key: true_target_idx, value: list of bool (True if covered at frame k)
    true_target_coverage_history = defaultdict(lambda: [False] * num_simulation_steps)

    # 记录每个已确认航迹在每一帧是否关联到真实目标
    # key: track_id, value: list of bool (True if associated with a true target at frame k)
    confirmed_track_association_history = defaultdict(lambda: [False] * num_simulation_steps)

    # 记录每个已确认航迹的生命周期 (出现帧, 消失帧或最后帧)
    # key: track_id, value: {'start_frame': k_start, 'end_frame': k_end, 'is_true_association_ever': False}
    track_lifecycles = {}

    total_confirmed_tracks_frames = 0 # 已确认航迹存在的总帧数
    total_associated_confirmed_tracks_frames = 0 # 已确认航迹且关联到真值的总帧数
    num_false_alarms_detected_by_dbscan = 0 # DBSCAN检测出的虚警簇数量（近似）
    num_true_detections_by_dbscan = 0 # DBSCAN检测出的真实目标簇数量（近似）

    # 记录每个真实目标的存在时间（帧数）
    true_target_lifetimes = [len(traj) for traj in true_target_trajs]


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

        # --- RMSE 和 指标数据收集 ---
        true_target_states_at_k_for_metrics = []
        for target_idx, true_traj_single_target in enumerate(true_target_trajs):
            if k < len(true_traj_single_target):
                true_target_states_at_k_for_metrics.append({
                    "id": target_idx, # 使用索引作为临时ID
                    "state_xy_vxvy": true_traj_single_target[k, :4],
                    "pos_xy": true_traj_single_target[k, :2]
                })

        confirmed_tracks_at_k_for_metrics = []
        active_track_ids_current_frame = set()
        for track_info in tracked_info_for_frame_k:
            active_track_ids_current_frame.add(track_info['id'])
            if track_info['id'] not in track_lifecycles:
                track_lifecycles[track_info['id']] = {'start_frame': k, 'end_frame': k, 'is_true_association_ever': False}
            else:
                track_lifecycles[track_info['id']]['end_frame'] = k

            if track_info['state'] == 'Confirmed' and track_info['history']:
                total_confirmed_tracks_frames += 1
                confirmed_tracks_at_k_for_metrics.append({
                    "id": track_info['id'],
                    "state_xy_vxvy": np.array(track_info['history'][-1][:4]),
                    "pos_xy": np.array(track_info['history'][-1][:2])
                })

        step_pos_sq_errors = []
        step_vel_sq_errors = []

        # --- DBSCAN 检测结果与真值关联 (用于估算 Pd_eff, Pfa_eff) ---
        # 注意: 这是一个简化的关联，实际Pd/Pfa计算更复杂
        # 这里我们只看DBSCAN输出的簇是否能关联到真实目标位置
        detected_cluster_associated_to_true = [False] * len(detected_clusters_info_k)
        if detected_clusters_info_k and true_target_states_at_k_for_metrics:
            num_det_clusters = len(detected_clusters_info_k)
            num_true_targets_curr = len(true_target_states_at_k_for_metrics)
            cost_matrix_dbscan_true = np.full((num_det_clusters, num_true_targets_curr), np.inf)
            for i, det_cluster in enumerate(detected_clusters_info_k):
                det_pos = np.array(det_cluster['position_global'])
                for j, true_target_data in enumerate(true_target_states_at_k_for_metrics):
                    true_pos = true_target_data["pos_xy"]
                    dist = np.linalg.norm(det_pos - true_pos)
                    cost_matrix_dbscan_true[i,j] = dist

            det_indices, true_indices = linear_sum_assignment(cost_matrix_dbscan_true)
            for d_idx, t_idx in zip(det_indices, true_indices):
                # 使用一个较宽松的门限来判断DBSCAN检测是否对应真实目标
                if cost_matrix_dbscan_true[d_idx, t_idx] < cfg.DBSCAN_EPS * 3:
                    detected_cluster_associated_to_true[d_idx] = True

        num_true_detections_by_dbscan += sum(detected_cluster_associated_to_true)
        num_false_alarms_detected_by_dbscan += (len(detected_clusters_info_k) - sum(detected_cluster_associated_to_true))


        # --- 已确认航迹与真值关联 (用于RMSE和航迹指标) ---
        if confirmed_tracks_at_k_for_metrics and true_target_states_at_k_for_metrics:
            num_confirmed_tracks = len(confirmed_tracks_at_k_for_metrics)
            num_true_targets_curr = len(true_target_states_at_k_for_metrics)
            cost_matrix_rmse = np.full((num_confirmed_tracks, num_true_targets_curr), np.inf)

            for i, confirmed_track_data in enumerate(confirmed_tracks_at_k_for_metrics):
                est_pos = confirmed_track_data["pos_xy"]
                for j, true_target_data in enumerate(true_target_states_at_k_for_metrics):
                    true_pos = true_target_data["pos_xy"]
                    dist = np.linalg.norm(est_pos - true_pos)
                    cost_matrix_rmse[i, j] = dist

            track_indices_rmse, true_target_indices_rmse = linear_sum_assignment(cost_matrix_rmse)

            for tr_idx_in_list, gt_idx_in_list in zip(track_indices_rmse, true_target_indices_rmse):
                if cost_matrix_rmse[tr_idx_in_list, gt_idx_in_list] < RMSE_ASSOCIATION_GATE:
                    confirmed_track_data = confirmed_tracks_at_k_for_metrics[tr_idx_in_list]
                    true_target_data = true_target_states_at_k_for_metrics[gt_idx_in_list]

                    est_state = confirmed_track_data["state_xy_vxvy"]
                    true_state = true_target_data["state_xy_vxvy"]

                    pos_error_sq = np.sum((est_state[:2] - true_state[:2])**2)
                    vel_error_sq = np.sum((est_state[2:4] - true_state[2:4])**2)
                    step_pos_sq_errors.append(pos_error_sq)
                    step_vel_sq_errors.append(vel_error_sq)

                    # 更新指标相关数据
                    true_target_coverage_history[true_target_data["id"]][k] = True
                    confirmed_track_association_history[confirmed_track_data["id"]][k] = True
                    track_lifecycles[confirmed_track_data["id"]]['is_true_association_ever'] = True
                    total_associated_confirmed_tracks_frames +=1

        all_time_steps_data_for_rmse.append({
            "time": current_time,
            "pos_sq_errors": step_pos_sq_errors,
            "vel_sq_errors": step_vel_sq_errors
        })

        if (k + 1) % 20 == 0 or k == num_simulation_steps - 1:
            num_raw = len(raw_measurements_polar_k_current_step)
            num_detected_clusters = len(detected_clusters_info_k)
            num_active_tracks_now = len(tracker.tracks) # tracker.tracks 包含所有活动航迹
            num_confirmed_now = sum(1 for t_info in tracked_info_for_frame_k if t_info['state'] == 'Confirmed')
            print(
                f"  Time {current_time:.1f}s: RawMeas={num_raw}, Detections={num_detected_clusters}, ActiveTracks={num_active_tracks_now} (Confirmed={num_confirmed_now})")

    # --- 计算最终的RMSE值 ---
    time_vector = [data['time'] for data in all_time_steps_data_for_rmse]
    position_rmse_values = []
    velocity_rmse_values = []

    for data in all_time_steps_data_for_rmse:
        if data['pos_sq_errors']:
            mean_pos_sq_error = np.mean(data['pos_sq_errors'])
            position_rmse_values.append(np.sqrt(mean_pos_sq_error))
        else:
            position_rmse_values.append(np.nan)
        if data['vel_sq_errors']:
            mean_vel_sq_error = np.mean(data['vel_sq_errors'])
            velocity_rmse_values.append(np.sqrt(mean_vel_sq_error))
        else:
            velocity_rmse_values.append(np.nan)

    # --- 计算其他性能指标 ---
    print("\n--- Performance Metrics ---")

    # 1. 平均RMSE
    avg_pos_rmse = np.nanmean(position_rmse_values) if np.any(np.isfinite(position_rmse_values)) else np.nan
    avg_vel_rmse = np.nanmean(velocity_rmse_values) if np.any(np.isfinite(velocity_rmse_values)) else np.nan
    print(f"Average Position RMSE: {avg_pos_rmse:.3f} m (over frames with valid associations)")
    print(f"Average Velocity RMSE: {avg_vel_rmse:.3f} m/s (over frames with valid associations)")

    # 2. 航迹完整度 (Track Completeness)
    total_target_existence_frames = sum(true_target_lifetimes)
    total_target_covered_frames = sum(sum(history) for history in true_target_coverage_history.values())
    avg_track_completeness = (total_target_covered_frames / total_target_existence_frames) * 100 if total_target_existence_frames > 0 else 0
    print(f"Average Track Completeness: {avg_track_completeness:.2f}%")

    # 3. 虚假航迹相关指标
    num_total_confirmed_tracks_generated = 0
    num_false_confirmed_tracks = 0
    for track_id, lifecycle_info in track_lifecycles.items():
        # 筛选出那些曾经达到Confirmed状态的航迹
        # (需要从 all_tracks_info_over_time 中确认该航迹是否曾达到Confirmed)
        # 这是一个简化：如果一个航迹在其生命周期内从未关联到真实目标，我们认为它是虚假航迹
        # 更准确的做法是检查其在Confirmed状态期间的关联情况
        is_ever_confirmed = False
        for frame_tracks in all_tracks_info_over_time:
            for track_data in frame_tracks:
                if track_data['id'] == track_id and track_data['state'] == 'Confirmed':
                    is_ever_confirmed = True
                    break
            if is_ever_confirmed:
                break

        if is_ever_confirmed:
            num_total_confirmed_tracks_generated += 1
            if not lifecycle_info['is_true_association_ever']:
                num_false_confirmed_tracks += 1

    false_track_rate_per_total_confirmed = (num_false_confirmed_tracks / num_total_confirmed_tracks_generated) * 100 if num_total_confirmed_tracks_generated > 0 else 0
    print(f"Total Confirmed Tracks Generated: {num_total_confirmed_tracks_generated}")
    print(f"Number of False Confirmed Tracks (never associated): {num_false_confirmed_tracks}")
    print(f"False Confirmed Track Rate: {false_track_rate_per_total_confirmed:.2f}%")

    # 另一种虚假航迹率：(总确认航迹帧数 - 总关联确认航迹帧数) / 总确认航迹帧数
    if total_confirmed_tracks_frames > 0:
        false_track_frame_ratio = ((total_confirmed_tracks_frames - total_associated_confirmed_tracks_frames) / total_confirmed_tracks_frames) * 100
        print(f"Ratio of Confirmed Track Frames that are False (unassociated): {false_track_frame_ratio:.2f}%")
    else:
        print("Ratio of Confirmed Track Frames that are False: N/A (no confirmed frames)")


    # 4. 航迹丢失率 (Track Loss Rate) - 这是一个比较复杂的指标，简化计算：
    # 考虑那些至少被跟踪了一段时间（比如超过N_CONFIRM_AGE帧）的真实目标，
    # 看它们在后续的生命周期中，航迹是否被删除了。
    # 这里的实现比较粗略：
    num_targets_potentially_lost = 0
    num_targets_actually_lost = 0
    min_tracking_duration_for_loss_eval = cfg.N_CONFIRM_AGE * 2 # 真实目标至少被跟踪这么久才考虑丢失

    for gt_idx in range(num_true_targets):
        gt_lifetime = true_target_lifetimes[gt_idx]
        if gt_lifetime < min_tracking_duration_for_loss_eval:
            continue # 目标存在时间太短，不评估丢失

        num_targets_potentially_lost += 1

        # 检查该真实目标是否在生命周期中段后丢失了覆盖
        # (即，在被覆盖一段时间后，后续帧不再被覆盖直到其生命周期结束)
        coverage = true_target_coverage_history[gt_idx]
        first_covered_frame = -1
        last_covered_frame = -1
        for frame_idx in range(gt_lifetime):
            if coverage[frame_idx]:
                if first_covered_frame == -1:
                    first_covered_frame = frame_idx
                last_covered_frame = frame_idx

        if first_covered_frame != -1 and (last_covered_frame < gt_lifetime - (cfg.MAX_CONSECUTIVE_MISSES // 2)) and (last_covered_frame - first_covered_frame > cfg.N_CONFIRM_AGE):
            # 如果最后覆盖帧远早于目标消失，且目标曾被稳定跟踪过一段时间
            num_targets_actually_lost +=1

    track_loss_rate = (num_targets_actually_lost / num_targets_potentially_lost) * 100 if num_targets_potentially_lost > 0 else 0
    print(f"Approx. Track Loss Rate: {track_loss_rate:.2f}% (for targets tracked > {min_tracking_duration_for_loss_eval*cfg.DT:.1f}s then lost)")

    # 5. 估算检测性能 (基于DBSCAN输出)
    # 这里的 "true detections" 是指DBSCAN的输出簇能关联到真实目标
    # "false alarms" 是指DBSCAN的输出簇不能关联到真实目标
    # 这不是严格意义上的雷达Pd, Pfa，而是DBSCAN处理后的结果
    avg_true_detections_per_frame = num_true_detections_by_dbscan / num_simulation_steps if num_simulation_steps > 0 else 0
    avg_false_alarms_per_frame_dbscan = num_false_alarms_detected_by_dbscan / num_simulation_steps if num_simulation_steps > 0 else 0
    print(f"Avg. DBSCAN True Detections per Frame (approx.): {avg_true_detections_per_frame:.2f}")
    print(f"Avg. DBSCAN False Alarms per Frame (approx.): {avg_false_alarms_per_frame_dbscan:.2f}")

    # 简单的检测概率估算：(DBSCAN检测到的真实目标数) / (场景中真实目标总数，假设都在传感器范围内)
    # 这是一个非常粗略的估计
    total_true_target_instances_in_sim = sum(true_target_lifetimes)
    if total_true_target_instances_in_sim > 0:
        overall_pd_estimate_dbscan = num_true_detections_by_dbscan / total_true_target_instances_in_sim
        print(f"Overall DBSCAN Detection Rate for True Targets (approx.): {overall_pd_estimate_dbscan:.2%}")


    print("\n--- Phase 5: Visualization ---")
    # ... (后续的绘图代码不变) ...
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
        time_vector=time_vector,
        position_rmse_over_time=position_rmse_values, # 使用新的变量名
        velocity_rmse_over_time=velocity_rmse_values  # 使用新的变量名
    )

    print("\nSimulation complete. Plot saved in 'output/plots/' directory.")

if __name__ == '__main__':
    run_simulation()
