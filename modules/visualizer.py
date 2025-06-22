# modules/visualizer.py
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec  # 确保导入
import numpy as np
import os
import config as cfg


def plot_results_with_roc(observer_trajectory, true_target_trajectories,
                          measurements_over_time_polar, detections_over_time_global_xy_list_of_lists,
                          all_tracks_info_over_time,
                          time_vector=None,  # 来自主仿真的RMSE数据
                          position_rmse_over_time=None,
                          velocity_rmse_over_time=None,
                          roc_data=None):  # 新增ROC数据参数
    """
    绘制所有仿真结果的最终图表，包括主轨迹、RMSE曲线和ROC曲线。
    """
    fig = plt.figure(figsize=(16, 24))
    fig.patch.set_facecolor('black')
    # 调整GridSpec为4行，并可能调整hspace
    gs = gridspec.GridSpec(4, 1, height_ratios=[2.5, 1, 1, 1.2], hspace=0.65)

    # --- 1. 主轨迹图 ---
    # (这部分与你提供的visualizer.py中的plot_results函数的主图绘制逻辑完全相同)
    ax_main = fig.add_subplot(gs[0, 0])
    ax_main.set_facecolor('black');
    ax_main.spines['bottom'].set_color('gray');
    ax_main.spines['top'].set_color('gray');
    ax_main.spines['right'].set_color('gray');
    ax_main.spines['left'].set_color('gray');
    ax_main.tick_params(axis='x', colors='gray');
    ax_main.tick_params(axis='y', colors='gray');
    ax_main.set_xlabel("X (m) - Global Coordinate System", color='white', fontsize=12);
    ax_main.set_ylabel("Y (m) - Global Coordinate System", color='white', fontsize=12);
    ax_main.set_title("Radar Simulation: Trajectories, Detections, and Tracks", color='white', fontsize=16);
    ax_main.grid(True, linestyle='--', alpha=0.2, color='gray');
    ax_main.axis('equal')
    all_meas_global_x, all_meas_global_y = [], [];
    for k, frame_polar_measurements in enumerate(measurements_over_time_polar):
        if k < len(observer_trajectory):
            obs_state_k = observer_trajectory[k];
            obs_x_g, obs_y_g, obs_vx_g, obs_vy_g = obs_state_k[:4]
            observer_heading_rad = np.arctan2(obs_vy_g, obs_vx_g) if not (
                        np.isclose(obs_vx_g, 0) and np.isclose(obs_vy_g, 0)) else 0.0
            for r, theta_local, _ in frame_polar_measurements:
                x_radar = r * np.cos(theta_local);
                y_radar = r * np.sin(theta_local)
                x_global_rel = x_radar * np.cos(observer_heading_rad) - y_radar * np.sin(observer_heading_rad);
                y_global_rel = x_radar * np.sin(observer_heading_rad) + y_radar * np.cos(observer_heading_rad)
                all_meas_global_x.append(obs_x_g + x_global_rel);
                all_meas_global_y.append(obs_y_g + y_global_rel)
    ax_main.scatter(all_meas_global_x, all_meas_global_y, color='dimgray', marker='x', s=15, alpha=0.4,
                    label='Raw Measurements (Global)')
    all_det_global_x, all_det_global_y = [], [];
    for frame_global_xy_detections in detections_over_time_global_xy_list_of_lists:
        for det_xy in frame_global_xy_detections: all_det_global_x.append(det_xy[0]); all_det_global_y.append(det_xy[1])
    ax_main.scatter(all_det_global_x, all_det_global_y, s=40, facecolors='none', edgecolors='magenta', alpha=0.9,
                    linewidths=1.5, label='DBSCAN Detections (Global)')
    ax_main.plot(observer_trajectory[:, 0], observer_trajectory[:, 1], color='cyan', linestyle='-',
                 label='Observer Trajectory', lw=2);
    ax_main.plot(observer_trajectory[0, 0], observer_trajectory[0, 1], marker='s', color='cyan', markersize=8,
                 label='Observer Start')
    for i, traj in enumerate(true_target_trajectories):
        color = cfg.TARGET_COLORS[i % len(cfg.TARGET_COLORS)];
        ax_main.plot(traj[:, 0], traj[:, 1], color=color, linestyle='-', label=f'True Target {i + 1}', lw=2.5);
        ax_main.plot(traj[0, 0], traj[0, 1], marker='o', color=color, markersize=8, label=f'Target {i + 1} Start')
    plotted_track_ids = set();
    if all_tracks_info_over_time and all_tracks_info_over_time[-1]:  # 使用最后一帧的航迹信息来绘制完整历史
        final_frame_tracks_info = all_tracks_info_over_time[-1]
        for track_info in final_frame_tracks_info:  # 遍历最后一帧的所有航迹
            track_id = track_info['id'];
            history_states = np.array(track_info['history']);
            track_state_status = track_info['state']
            color_index = (track_id - 1) % len(cfg.TARGET_COLORS);
            color = cfg.TARGET_COLORS[color_index]
            if history_states.shape[0] > 1:
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
    legend_main = ax_main.legend(facecolor='black', framealpha=0.8, edgecolor='gray', loc='upper left', fontsize=10);
    [text.set_color('white') for text in legend_main.get_texts()]

    # --- 2. 位置RMSE 图 ---
    # (与你提供的visualizer.py中plot_results的RMSE图绘制逻辑相同，但使用传入的RMSE数据)
    ax_pos_rmse = fig.add_subplot(gs[1, 0])
    if time_vector is not None and position_rmse_over_time is not None and len(time_vector) == len(
            position_rmse_over_time):
        ax_pos_rmse.plot(time_vector, position_rmse_over_time, color='lime', lw=1.5, label='Position RMSE')
    ax_pos_rmse.set_title('Position RMSE over Time', color='white', fontsize=14)  # 简化标题
    ax_pos_rmse.set_xlabel('Time (s)', color='white');
    ax_pos_rmse.set_ylabel('Position RMSE (m)', color='white')
    ax_pos_rmse.grid(True, linestyle='--', alpha=0.2, color='gray');
    ax_pos_rmse.set_facecolor('black')
    ax_pos_rmse.tick_params(axis='x', colors='gray');
    ax_pos_rmse.tick_params(axis='y', colors='gray');
    [spine.set_color('gray') for spine in ax_pos_rmse.spines.values()]
    if time_vector is not None and position_rmse_over_time is not None and any(
            not np.isnan(val) for val in position_rmse_over_time if val is not None):
        legend_pos = ax_pos_rmse.legend(facecolor='black', framealpha=0.8, edgecolor='gray', loc='upper right',
                                        fontsize=9)
        for text_l in legend_pos.get_texts(): text_l.set_color('white')
    else:
        ax_pos_rmse.text(0.5, 0.5, 'No Position RMSE Data', ha='center', va='center', transform=ax_pos_rmse.transAxes,
                         color='gray', fontsize=10)

    # --- 3. 速度RMSE 图 ---
    ax_vel_rmse = fig.add_subplot(gs[2, 0])
    if time_vector is not None and velocity_rmse_over_time is not None and len(time_vector) == len(
            velocity_rmse_over_time):
        ax_vel_rmse.plot(time_vector, velocity_rmse_over_time, color='yellow', lw=1.5, label='Velocity RMSE')
    ax_vel_rmse.set_title('Velocity RMSE over Time', color='white', fontsize=14)  # 简化标题
    ax_vel_rmse.set_xlabel('Time (s)', color='white');
    ax_vel_rmse.set_ylabel('Velocity RMSE (m/s)', color='white')
    ax_vel_rmse.grid(True, linestyle='--', alpha=0.2, color='gray');
    ax_vel_rmse.set_facecolor('black')
    ax_vel_rmse.tick_params(axis='x', colors='gray');
    ax_vel_rmse.tick_params(axis='y', colors='gray');
    [spine.set_color('gray') for spine in ax_vel_rmse.spines.values()]
    if time_vector is not None and velocity_rmse_over_time is not None and any(
            not np.isnan(val) for val in velocity_rmse_over_time if val is not None):
        legend_vel = ax_vel_rmse.legend(facecolor='black', framealpha=0.8, edgecolor='gray', loc='upper right',
                                        fontsize=9)
        for text_l in legend_vel.get_texts(): text_l.set_color('white')
    else:
        ax_vel_rmse.text(0.5, 0.5, 'No Velocity RMSE Data', ha='center', va='center', transform=ax_vel_rmse.transAxes,
                         color='gray', fontsize=10)

    # --- 4. ROC 图 ---
    ax_roc = fig.add_subplot(gs[3, 0])
    if roc_data:  # roc_data 是一个列表，每个元素是 {'eps': ..., 'pfa': ..., 'pd': ...}
        pfa_values = [item['pfa'] for item in roc_data]
        pd_values = [item['pd'] for item in roc_data]

        ax_roc.plot(pfa_values, pd_values, marker='o', color='deepskyblue', linestyle='-', lw=1.5,
                    label='ROC Curve (varying DBSCAN_EPS)')

        min_pfa_for_diag, max_pfa_for_diag = 0, 0.01  # Default for diagonal
        valid_pfa = [p for p in pfa_values if isinstance(p, (int, float)) and not np.isnan(p)]
        if valid_pfa: max_pfa_for_diag = max(valid_pfa) if valid_pfa else 0.01
        ax_roc.plot([min_pfa_for_diag, max_pfa_for_diag], [min_pfa_for_diag, max_pfa_for_diag], linestyle='--',
                    color='red', lw=1, label='Random Guess')

        ax_roc.set_title('Receiver Operating Characteristic (ROC) Curve', color='white', fontsize=14)
        ax_roc.set_xlabel('False Alarm Probability (P_fa)', color='white')
        ax_roc.set_ylabel('Detection Probability (P_d)', color='white')
        ax_roc.grid(True, linestyle='--', alpha=0.2, color='gray');
        ax_roc.set_facecolor('black')
        ax_roc.tick_params(axis='x', colors='gray', labelrotation=30);
        ax_roc.tick_params(axis='y', colors='gray')

        if valid_pfa:
            min_val_pfa, max_val_pfa = min(valid_pfa), max(valid_pfa)
            # Add a small epsilon if min and max are the same to avoid zero range
            x_left = min_val_pfa - 0.01 * abs(max_val_pfa) if max_val_pfa != min_val_pfa else min_val_pfa - 0.0001
            x_right = max_val_pfa * 1.1 if max_val_pfa > 0 else (0.001 if min_val_pfa == 0 else max_val_pfa + 0.0001)
            if x_left >= x_right: x_right = x_left + 0.0001  # Ensure right > left
            ax_roc.set_xlim(left=x_left, right=x_right)
            if max_val_pfa < 0.01 and max_val_pfa > 0: ax_roc.xaxis.set_major_formatter(plt.FormatStrFormatter('%.1e'))
        else:
            ax_roc.set_xlim(left=0, right=0.01)  # Default if no valid pfa data
        ax_roc.set_ylim(bottom=0, top=1.05)
        for spine in ax_roc.spines.values(): spine.set_color('gray')
        legend_roc = ax_roc.legend(facecolor='black', framealpha=0.8, edgecolor='gray', loc='lower right', fontsize=9)
        for text_l in legend_roc.get_texts(): text_l.set_color('white')
    else:
        ax_roc.text(0.5, 0.5, 'No ROC Data', ha='center', va='center', transform=ax_roc.transAxes, color='gray',
                    fontsize=12)
        ax_roc.set_facecolor('black');
        ax_roc.grid(True, linestyle='--', alpha=0.2, color='gray')
        ax_roc.tick_params(axis='x', colors='gray');
        ax_roc.tick_params(axis='y', colors='gray')
        for spine in ax_roc.spines.values(): spine.set_color('gray')
        ax_roc.set_xlabel('False Alarm Probability (P_fa)', color='white')
        ax_roc.set_ylabel('Detection Probability (P_d)', color='white')
        ax_roc.set_title('Receiver Operating Characteristic (ROC) Curve', color='white', fontsize=14)

    fig.suptitle("Radar Simulation Analysis", color='white', fontsize=18, y=0.995)
    plot_path = os.path.join(cfg.OUTPUT_PLOTS_DIR, "full_analysis_with_roc.png")  # 新的文件名
    plt.savefig(plot_path, facecolor='black', dpi=200, bbox_inches='tight')
    print(f"\nFull analysis plot with ROC saved to: {plot_path}")
    plt.show()