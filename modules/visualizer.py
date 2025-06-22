# modules/visualizer.py
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
import os
import config as cfg  # 确保config.py在可访问的路径


def plot_results_with_roc(observer_trajectory, true_target_trajectories,
                          measurements_over_time_polar, detections_over_time_global_xy_list_of_lists,
                          all_tracks_info_over_time,
                          time_vector=None,
                          position_rmse_over_time=None,
                          velocity_rmse_over_time=None,
                          roc_curve_points=None):
    """
    绘制所有仿真结果的最终图表，包括主轨迹、RMSE曲线和ROC曲线。
    """
    fig = plt.figure(figsize=(16, 24))  # 调整高度以容纳ROC图
    fig.patch.set_facecolor('black')

    # 使用 GridSpec 来布局子图
    # 4行: 主轨迹图, 位置RMSE图, 速度RMSE图, ROC图
    gs = gridspec.GridSpec(4, 1, height_ratios=[2.5, 1, 1, 1.2], hspace=0.6)  # 稍微增加垂直间距

    # --- 1. 主轨迹图 ---
    ax_main = fig.add_subplot(gs[0, 0])
    ax_main.set_facecolor('black')
    ax_main.spines['bottom'].set_color('gray')
    ax_main.spines['top'].set_color('gray')
    ax_main.spines['right'].set_color('gray')
    ax_main.spines['left'].set_color('gray')
    ax_main.tick_params(axis='x', colors='gray')
    ax_main.tick_params(axis='y', colors='gray')
    ax_main.set_xlabel("X (m) - Global Coordinate System", color='white', fontsize=12)
    ax_main.set_ylabel("Y (m) - Global Coordinate System", color='white', fontsize=12)
    ax_main.set_title("Radar Simulation: Trajectories, Detections, and Tracks", color='white', fontsize=16)
    ax_main.grid(True, linestyle='--', alpha=0.2, color='gray')
    ax_main.axis('equal')

    # 1.1 绘制原始雷达观测点 (灰色x)
    all_meas_global_x, all_meas_global_y = [], []
    for k, frame_polar_measurements in enumerate(measurements_over_time_polar):
        if k < len(observer_trajectory):
            obs_state_k = observer_trajectory[k]
            obs_x_g, obs_y_g, obs_vx_g, obs_vy_g = obs_state_k[:4]
            if not (np.isclose(obs_vx_g, 0) and np.isclose(obs_vy_g, 0)):
                observer_heading_rad = np.arctan2(obs_vy_g, obs_vx_g)
            else:
                observer_heading_rad = 0.0

            for r, theta_local, _ in frame_polar_measurements:
                x_radar = r * np.cos(theta_local)
                y_radar = r * np.sin(theta_local)
                x_global_rel = x_radar * np.cos(observer_heading_rad) - y_radar * np.sin(observer_heading_rad)
                y_global_rel = x_radar * np.sin(observer_heading_rad) + y_radar * np.cos(observer_heading_rad)
                all_meas_global_x.append(obs_x_g + x_global_rel)
                all_meas_global_y.append(obs_y_g + y_global_rel)
    ax_main.scatter(all_meas_global_x, all_meas_global_y, color='dimgray', marker='x', s=15, alpha=0.4,
                    label='Raw Measurements (Global)')

    # 1.2 绘制DBSCAN检测结果 (紫色空心圆圈)
    all_det_global_x, all_det_global_y = [], []
    for frame_global_xy_detections in detections_over_time_global_xy_list_of_lists:
        for det_xy in frame_global_xy_detections:
            all_det_global_x.append(det_xy[0])
            all_det_global_y.append(det_xy[1])
    ax_main.scatter(all_det_global_x, all_det_global_y, s=40, facecolors='none', edgecolors='magenta', alpha=0.9,
                    linewidths=1.5, label='DBSCAN Detections (Global)')

    # 1.3 绘制观测者轨迹 (蓝色实线)
    ax_main.plot(observer_trajectory[:, 0], observer_trajectory[:, 1], color='cyan', linestyle='-',
                 label='Observer Trajectory', lw=2)
    ax_main.plot(observer_trajectory[0, 0], observer_trajectory[0, 1], marker='s', color='cyan', markersize=8,
                 label='Observer Start')

    # 1.4 绘制真实目标轨迹 (彩色实线)
    for i, traj in enumerate(true_target_trajectories):
        color = cfg.TARGET_COLORS[i % len(cfg.TARGET_COLORS)]
        ax_main.plot(traj[:, 0], traj[:, 1], color=color, linestyle='-', label=f'True Target {i + 1}', lw=2.5)
        ax_main.plot(traj[0, 0], traj[0, 1], marker='o', color=color, markersize=8, label=f'Target {i + 1} Start')

    # 1.5 绘制最终的跟踪轨迹 (带ID的彩色虚线)
    plotted_track_ids = set()
    if all_tracks_info_over_time and all_tracks_info_over_time[-1]:  # Use tracks from the last frame for labeling
        final_frame_tracks_info = all_tracks_info_over_time[-1]
        # 先绘制所有航迹历史，再在最后点标记ID，避免图例重复
        all_track_histories_to_plot = {}  # track_id -> {'history': [], 'state': '', 'color': ''}
        for frame_idx, frame_tracks_info in enumerate(all_tracks_info_over_time):
            for track_info in frame_tracks_info:
                track_id = track_info['id']
                if track_id not in all_track_histories_to_plot:
                    color_index = (track_id - 1) % len(cfg.TARGET_COLORS)
                    all_track_histories_to_plot[track_id] = {
                        'history_x': [],
                        'history_y': [],
                        'state': track_info['state'],  # Store last known state
                        'color': cfg.TARGET_COLORS[color_index],
                        'last_pos': None
                    }
                if track_info['history']:  # 确保历史非空
                    # 只添加当前帧的最新点到历史
                    current_pos = track_info['history'][-1]  # 这是整个历史的最后点
                    # 我们需要的是当前帧的估计点，这应该从tracker.step的返回中获取
                    # 假设 track_info['history'] 存储的是到当前帧为止的完整历史
                    # 那么我们直接用这个历史的最后点作为当前帧的点
                    if len(track_info['history']) > frame_idx:  # 确保历史长度足够
                        # This logic is a bit tricky if history is cumulative.
                        # For simplicity, we'll plot based on the final_frame_tracks_info's history later.
                        pass  # We will plot full histories from final_frame_tracks_info

        # 使用最后一帧的航迹信息来绘制完整的历史轨迹
        for track_info in final_frame_tracks_info:
            track_id = track_info['id']
            history_states = np.array(track_info['history'])  # This is the full history up to the end
            track_state_status = track_info['state']  # State at the end of simulation
            color_index = (track_id - 1) % len(cfg.TARGET_COLORS)
            color = cfg.TARGET_COLORS[color_index]

            if history_states.shape[0] > 1:  # 至少有两个点才能画线
                label_text = ""
                if track_state_status == 'Confirmed':
                    label_text = f'Est. Track {track_id}' if track_id not in plotted_track_ids else "_nolegend_"
                    ax_main.plot(history_states[:, 0], history_states[:, 1], color=color, linestyle='--', marker='.',
                                 markersize=4, lw=2.0, label=label_text)
                    ax_main.text(history_states[-1, 0] + 1, history_states[-1, 1] + 1, f'ID:{track_id}', color=color,
                                 fontsize=10, weight='bold')
                elif track_state_status == 'Tentative':
                    label_text = f'Tent. Track {track_id}' if track_id not in plotted_track_ids else "_nolegend_"
                    ax_main.plot(history_states[:, 0], history_states[:, 1], color=color, linestyle=':', marker='.',
                                 markersize=2, lw=1.0, alpha=0.6, label=label_text)
                plotted_track_ids.add(track_id)

    legend_main = ax_main.legend(facecolor='black', framealpha=0.8, edgecolor='gray', loc='upper left', fontsize=10)
    for text in legend_main.get_texts():
        text.set_color('white')

    # --- 2. 位置RMSE 图 ---
    if time_vector is not None and position_rmse_over_time is not None:
        ax_pos_rmse = fig.add_subplot(gs[1, 0])
        ax_pos_rmse.plot(time_vector, position_rmse_over_time, color='lime', lw=1.5, label='Position RMSE')
        ax_pos_rmse.set_title('Position RMSE over Time (Confirmed Tracks vs True Targets)', color='white', fontsize=14)
        ax_pos_rmse.set_xlabel('Time (s)', color='white')
        ax_pos_rmse.set_ylabel('Position RMSE (m)', color='white')
        ax_pos_rmse.grid(True, linestyle='--', alpha=0.2, color='gray')
        ax_pos_rmse.set_facecolor('black')
        ax_pos_rmse.tick_params(axis='x', colors='gray')
        ax_pos_rmse.tick_params(axis='y', colors='gray')
        for spine in ax_pos_rmse.spines.values(): spine.set_color('gray')
        if any(not np.isnan(val) for val in position_rmse_over_time if val is not None):
            legend_pos = ax_pos_rmse.legend(facecolor='black', framealpha=0.8, edgecolor='gray', loc='upper right',
                                            fontsize=9)
            for text_l in legend_pos.get_texts(): text_l.set_color('white')

    # --- 3. 速度RMSE 图 ---
    if time_vector is not None and velocity_rmse_over_time is not None:
        ax_vel_rmse = fig.add_subplot(gs[2, 0])
        ax_vel_rmse.plot(time_vector, velocity_rmse_over_time, color='yellow', lw=1.5, label='Velocity RMSE')
        ax_vel_rmse.set_title('Velocity RMSE over Time (Confirmed Tracks vs True Targets)', color='white', fontsize=14)
        ax_vel_rmse.set_xlabel('Time (s)', color='white')
        ax_vel_rmse.set_ylabel('Velocity RMSE (m/s)', color='white')
        ax_vel_rmse.grid(True, linestyle='--', alpha=0.2, color='gray')
        ax_vel_rmse.set_facecolor('black')
        ax_vel_rmse.tick_params(axis='x', colors='gray')
        ax_vel_rmse.tick_params(axis='y', colors='gray')
        for spine in ax_vel_rmse.spines.values(): spine.set_color('gray')
        if any(not np.isnan(val) for val in velocity_rmse_over_time if val is not None):
            legend_vel = ax_vel_rmse.legend(facecolor='black', framealpha=0.8, edgecolor='gray', loc='upper right',
                                            fontsize=9)
            for text_l in legend_vel.get_texts(): text_l.set_color('white')

    # --- 4. ROC 图 ---
    if roc_curve_points:
        ax_roc = fig.add_subplot(gs[3, 0])
        pfa_values = [p[0] for p in roc_curve_points]
        pd_values = [p[1] for p in roc_curve_points]

        ax_roc.plot(pfa_values, pd_values, marker='o', color='deepskyblue', linestyle='-', lw=1.5,
                    label='ROC Curve (varying DBSCAN_EPS)')

        min_pfa_for_diag = 0
        max_pfa_for_diag = 0
        if pfa_values:  # Ensure pfa_values is not empty
            # Filter out potential None or non-numeric values if any, though sort should handle it
            valid_pfa = [p for p in pfa_values if isinstance(p, (int, float)) and not np.isnan(p)]
            if valid_pfa:
                max_pfa_for_diag = max(valid_pfa) if valid_pfa else 0.01
            else:  # if no valid pfa values
                max_pfa_for_diag = 0.01
        else:  # if pfa_values is empty
            max_pfa_for_diag = 0.01

        ax_roc.plot([min_pfa_for_diag, max_pfa_for_diag], [min_pfa_for_diag, max_pfa_for_diag], linestyle='--',
                    color='red', lw=1, label='Random Guess')

        ax_roc.set_title('Receiver Operating Characteristic (ROC) Curve', color='white', fontsize=14)
        ax_roc.set_xlabel('False Alarm Probability (P_fa)', color='white')
        ax_roc.set_ylabel('Detection Probability (P_d)', color='white')
        ax_roc.grid(True, linestyle='--', alpha=0.2, color='gray')
        ax_roc.set_facecolor('black')
        ax_roc.tick_params(axis='x', colors='gray', labelrotation=30)
        ax_roc.tick_params(axis='y', colors='gray')

        if pfa_values and valid_pfa:  # Check valid_pfa again
            # Ensure min and max are calculated on valid_pfa
            min_val_pfa = min(valid_pfa)
            max_val_pfa = max(valid_pfa)
            ax_roc.set_xlim(left=min_val_pfa - 0.01 * max_val_pfa if max_val_pfa > 0 else -0.001,
                            right=max_val_pfa * 1.1 if max_val_pfa > 0 else 0.01)
        else:  # Default if no valid pfa data
            ax_roc.set_xlim(left=0, right=0.1)

        ax_roc.set_ylim(bottom=0, top=1.05)

        if pfa_values and valid_pfa and max(valid_pfa) < 0.01 and max(
                valid_pfa) > 0:  # Check valid_pfa and positive max
            ax_roc.xaxis.set_major_formatter(plt.FormatStrFormatter('%.1e'))

        for spine in ax_roc.spines.values(): spine.set_color('gray')
        legend_roc = ax_roc.legend(facecolor='black', framealpha=0.8, edgecolor='gray', loc='lower right', fontsize=9)
        for text_l in legend_roc.get_texts(): text_l.set_color('white')

    fig.suptitle("Radar Simulation Analysis", color='white', fontsize=18, y=0.995)  # y调整避免与子图标题重叠

    plot_path = os.path.join(cfg.OUTPUT_PLOTS_DIR, "full_analysis_with_roc.png")
    plt.savefig(plot_path, facecolor='black', dpi=200, bbox_inches='tight')
    print(f"\nFull analysis plot with ROC saved to: {plot_path}")

    plt.show()