# modules/visualizer.py
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
import config as cfg
import matplotlib
import os

# ... (plot_initial_scene 函数保持不变) ...

def plot_trajectories_and_measurements(
    observer_trajectory,
    true_target_trajectories,
    radar_measurements_over_time,
    detected_points_global_over_time=None, # 可以用来画DBSCAN的输出（如果需要区分）
    estimated_target_trajectories_over_time=None, # 改为接收每一帧的航迹列表
    plot_observer_shape_interval=0 # 默认不画车的形状，只画轨迹
):
    fig, ax = plt.subplots(figsize=(12, 9))
    ax.set_xlabel("X (m) - Global Coordinate System (Forward)")
    ax.set_ylabel("Y (m) - Global Coordinate System (Left)")
    ax.set_title("Trajectories, Measurements, Detections, and Tracks")
    ax.grid(True, linestyle='--', alpha=0.7)
    ax.axis('equal')

    ax.plot(observer_trajectory[:, 0], observer_trajectory[:, 1], color=cfg.OBSERVER_COLOR, linestyle='-', label=cfg.OBSERVER_LABEL, lw=1.5, alpha=0.7)
    ax.plot(observer_trajectory[0, 0], observer_trajectory[0, 1], marker='o', color=cfg.OBSERVER_COLOR, markersize=8, label='_nolegend_')

    if plot_observer_shape_interval and plot_observer_shape_interval > 0:
        for k in range(0, len(observer_trajectory), plot_observer_shape_interval):
            # ... (绘制车辆形状的代码，如果需要可以取消注释) ...
            pass


    for i, traj in enumerate(true_target_trajectories):
        ax.plot(traj[:, 0], traj[:, 1], '-', color=cfg.TARGET_COLORS[i], label=f"True {cfg.TARGET_LABELS[i]}", lw=2, alpha=0.8)
        ax.plot(traj[0, 0], traj[0, 1], 'o', color=cfg.TARGET_COLORS[i], markersize=8, label='_nolegend_')

    all_meas_x, all_meas_y = [], []
    for k, step_measurements in enumerate(radar_measurements_over_time):
        if step_measurements:
            obs_x, obs_y = observer_trajectory[k, 0], observer_trajectory[k, 1]
            obs_vx, obs_vy = observer_trajectory[k, 2], observer_trajectory[k, 3]
            observer_heading_rad = np.arctan2(obs_vy, obs_vx) if not (np.isclose(obs_vx, 0) and np.isclose(obs_vy, 0)) else 0
            for r, theta, _ in step_measurements:
                x_radar_coord = r * np.cos(theta)
                y_radar_coord = r * np.sin(theta)
                x_global_rel = x_radar_coord * np.cos(observer_heading_rad) - y_radar_coord * np.sin(observer_heading_rad)
                y_global_rel = x_radar_coord * np.sin(observer_heading_rad) + y_radar_coord * np.cos(observer_heading_rad)
                x_global_abs = obs_x + x_global_rel
                y_global_abs = obs_y + y_global_rel
                all_meas_x.append(x_global_abs)
                all_meas_y.append(y_global_abs)
    if all_meas_x:
        ax.scatter(all_meas_x, all_meas_y, color='lightgray', marker='x', s=10, alpha=0.3, label='Raw Measurements')

    # 绘制DBSCAN检测到的点 (如果传入了)
    if detected_points_global_over_time: # 假设这是 all_detected_points_global_over_time
        all_dbscan_det_x, all_dbscan_det_y = [], []
        for step_detections in detected_points_global_over_time:
            if step_detections:
                for pt in step_detections:
                    all_dbscan_det_x.append(pt[0])
                    all_dbscan_det_y.append(pt[1])
        if all_dbscan_det_x:
            ax.scatter(all_dbscan_det_x, all_dbscan_det_y, color='purple', marker='o', s=20, alpha=0.5,
                       facecolors='none', edgecolors='purple', label='DBSCAN Detections (Clusters)')

    # 绘制估计的航迹历史
    if estimated_target_trajectories_over_time:
        # 需要从 all_steps_estimated_tracks_history 中提取每个独立航迹的完整历史
        # 当前 all_steps_estimated_tracks_history 是一个列表，每个元素是该帧的航迹信息列表
        # 我们需要重新组织它，按track_id聚合历史
        plotted_track_ids = set()
        final_tracks_data = {} # {track_id: [[x,y,vx,vy], ...]}

        for frame_tracks in estimated_target_trajectories_over_time:
            for track_info in frame_tracks:
                tid = track_info['id']
                if tid not in final_tracks_data:
                    final_tracks_data[tid] = {'history': [], 'color': track_info['color'], 'final_state': track_info['state']}
                # 只添加历史的最后一点，因为tracker.step返回的是当前帧的航迹状态列表，
                # 而track对象内部的history是完整的。我们应该从track对象获取完整历史。
                # 为了简化，这里假设estimated_target_trajectories_over_time的每个元素
                # 已经是某个航迹在所有时刻的完整历史列表。
                # 但根据tracker.step的返回，它返回的是当前帧的航迹状态。
                # 所以，我们需要在main.py中收集每个航迹的完整历史。

        # 假设 estimated_target_trajectories_over_time 已经是整理好的:
        # list of dictionaries, where each dict is {'id': id, 'history': [[px,py,vx,vy],...], 'color': color}
        # 并且这个history是该航迹从出现到当前帧的完整历史
        # 为了绘图，我们需要在main.py中收集每个航迹的最终完整历史
        
        # 临时的绘制方法：绘制每一帧的估计点，颜色根据ID
        # 这会导致一个航迹被画很多次，但能看出效果
        # 更好的方法是在main.py中收集每个航迹的完整历史，然后一次性画出
        for k, frame_tracks in enumerate(estimated_target_trajectories_over_time):
            for track_info in frame_tracks:
                track_id = track_info['id']
                # state_global = track_info['state'] # 这是当前帧的状态
                # ax.plot(state_global[0], state_global[1], 'o', color=track_info['color'], markersize=3, alpha=0.6, label=f'Est. Track {track_id}' if track_id not in plotted_track_ids else "_nolegend_")
                # plotted_track_ids.add(track_id)

                # 改为绘制航迹历史
                history_np = np.array(track_info['history'])
                if history_np.ndim == 2 and history_np.shape[0] > 1: # 确保有历史点且至少两个点才能画线
                    if track_id not in plotted_track_ids:
                        ax.plot(history_np[:, 0], history_np[:, 1], '--', color=track_info['color'], linewidth=1.5, label=f'Est. Track {track_id}')
                        plotted_track_ids.add(track_id)
                    else: # 对于同一航迹的后续帧，只更新最后一个点，避免重复图例
                        ax.plot(history_np[:, 0], history_np[:, 1], '--', color=track_info['color'], linewidth=1.5, label="_nolegend_")


    ax.set_xlim(cfg.PLOT_AREA_X_MIN, cfg.PLOT_AREA_X_MAX)
    ax.set_ylim(cfg.PLOT_AREA_Y_MIN, cfg.PLOT_AREA_Y_MAX)
    handles, labels = ax.get_legend_handles_labels()
    # 去重图例
    by_label = dict(zip(labels, handles))
    ax.legend(by_label.values(), by_label.keys(), loc='upper left', bbox_to_anchor=(1.02, 1), borderaxespad=0., fontsize='small')
    plt.tight_layout(rect=[0, 0, 0.80, 1]) # 给图例留更多空间

    os.makedirs(cfg.OUTPUT_PLOTS_DIR, exist_ok=True)
    plt.savefig(os.path.join(cfg.OUTPUT_PLOTS_DIR, "trajectories_measurements_detections_tracks.png"))
    print(f"Plot saved to {os.path.join(cfg.OUTPUT_PLOTS_DIR, 'trajectories_measurements_detections_tracks.png')}")
