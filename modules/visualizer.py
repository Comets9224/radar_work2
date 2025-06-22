# modules/visualizer.py
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec  # <--- 新增导入
import numpy as np
import os
import config as cfg


def plot_results(observer_trajectory, true_target_trajectories,
                 measurements_over_time_polar, detections_over_time_global_xy_list_of_lists,
                 all_tracks_info_over_time,
                 # 新增参数用于RMSE绘图
                 time_vector=None,
                 position_rmse_over_time=None,
                 velocity_rmse_over_time=None):
    """
    绘制所有仿真结果的最终图表，包括RMSE曲线。
    ... (其他参数说明保持不变) ...
    :param time_vector: 用于RMSE图的时间轴列表
    :param position_rmse_over_time: 每个时间点的位置RMSE列表
    :param velocity_rmse_over_time: 每个时间点的速度RMSE列表
    """
    # fig, ax = plt.subplots(figsize=(16, 12)) # 旧的单一图设置

    # 创建一个更大的图形窗口，为RMSE子图留出空间
    fig = plt.figure(figsize=(16, 18))  # 调整高度以容纳更多子图
    fig.patch.set_facecolor('black')

    # 使用 GridSpec 来布局子图
    # 3行: 主轨迹图 (占据较多空间), 位置RMSE图, 速度RMSE图
    # height_ratios 控制行高比例, hspace 控制子图垂直间距
    gs = gridspec.GridSpec(3, 1, height_ratios=[2.5, 1, 1], hspace=0.45)

    ax_main = fig.add_subplot(gs[0, 0])  # 主轨迹图的Axes对象

    # --- 主轨迹图的绘制代码 (基本不变, 'ax' 替换为 'ax_main') ---
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

    # 1. 绘制原始雷达观测点 (灰色x)
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

    # 2. 绘制DBSCAN检测结果 (紫色空心圆圈)
    all_det_global_x, all_det_global_y = [], []
    for frame_global_xy_detections in detections_over_time_global_xy_list_of_lists:
        for det_xy in frame_global_xy_detections:
            all_det_global_x.append(det_xy[0])
            all_det_global_y.append(det_xy[1])
    ax_main.scatter(all_det_global_x, all_det_global_y, s=40, facecolors='none', edgecolors='magenta', alpha=0.9,
                    linewidths=1.5, label='DBSCAN Detections (Global)')

    # 3. 绘制观测者轨迹 (蓝色实线)
    ax_main.plot(observer_trajectory[:, 0], observer_trajectory[:, 1], color='cyan', linestyle='-',
                 label='Observer Trajectory', lw=2)
    ax_main.plot(observer_trajectory[0, 0], observer_trajectory[0, 1], marker='s', color='cyan', markersize=8,
                 label='Observer Start')

    # 4. 绘制真实目标轨迹 (彩色实线)
    for i, traj in enumerate(true_target_trajectories):
        color = cfg.TARGET_COLORS[i % len(cfg.TARGET_COLORS)]
        ax_main.plot(traj[:, 0], traj[:, 1], color=color, linestyle='-', label=f'True Target {i + 1}', lw=2.5)
        ax_main.plot(traj[0, 0], traj[0, 1], marker='o', color=color, markersize=8, label=f'Target {i + 1} Start')

    # 5. 绘制最终的跟踪轨迹 (带ID的彩色虚线)
    plotted_track_ids = set()
    if all_tracks_info_over_time and all_tracks_info_over_time[-1]:
        final_frame_tracks_info = all_tracks_info_over_time[-1]
        for track_info in final_frame_tracks_info:
            track_id = track_info['id']
            history_states = np.array(track_info['history'])
            track_state_status = track_info['state']

            color_index = (track_id - 1) % len(cfg.TARGET_COLORS)  # 使用航迹ID来决定颜色
            color = cfg.TARGET_COLORS[color_index]

            if history_states.shape[0] > 1 and track_state_status == 'Confirmed':
                label_text = f'Est. Track {track_id}' if track_id not in plotted_track_ids else "_nolegend_"
                ax_main.plot(history_states[:, 0], history_states[:, 1], color=color, linestyle='--', marker='.',
                             markersize=4, lw=2.0, label=label_text)
                ax_main.text(history_states[-1, 0] + 1, history_states[-1, 1] + 1, f'ID:{track_id}', color=color,
                             fontsize=10, weight='bold')
                plotted_track_ids.add(track_id)
            elif history_states.shape[0] > 1 and track_state_status == 'Tentative':
                label_text = f'Tent. Track {track_id}' if track_id not in plotted_track_ids else "_nolegend_"
                ax_main.plot(history_states[:, 0], history_states[:, 1], color=color, linestyle=':', marker='.',
                             markersize=2, lw=1.0, alpha=0.6, label=label_text)
                plotted_track_ids.add(track_id)

    legend_main = ax_main.legend(facecolor='black', framealpha=0.8, edgecolor='gray', loc='upper left', fontsize=10)
    for text in legend_main.get_texts():
        text.set_color('white')
    # --- 主轨迹图绘制结束 ---

    # --- RMSE 图绘制 ---
    if time_vector is not None and position_rmse_over_time is not None and velocity_rmse_over_time is not None:
        # 位置RMSE图
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
        if any(not np.isnan(val) for val in position_rmse_over_time):  # 只有在有数据时才显示图例
            legend_pos = ax_pos_rmse.legend(facecolor='black', framealpha=0.8, edgecolor='gray', loc='upper right',
                                            fontsize=9)
            for text in legend_pos.get_texts(): text.set_color('white')

        # 速度RMSE图
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
        if any(not np.isnan(val) for val in velocity_rmse_over_time):  # 只有在有数据时才显示图例
            legend_vel = ax_vel_rmse.legend(facecolor='black', framealpha=0.8, edgecolor='gray', loc='upper right',
                                            fontsize=9)
            for text in legend_vel.get_texts(): text.set_color('white')
    # --- RMSE 图绘制结束 ---

    # plt.tight_layout(rect=[0, 0, 1, 0.97]) # 调整布局，rect的最后一个参数可能需要微调以避免标题被fig.suptitle遮挡
    fig.suptitle("Radar Simulation Analysis", color='white', fontsize=18, y=0.99)  # 总标题

    plot_path = os.path.join(cfg.OUTPUT_PLOTS_DIR, "final_tracking_result_with_rmse.png")  # 新的文件名
    plt.savefig(plot_path, facecolor='black', dpi=200, bbox_inches='tight')
    print(f"\nFinal plot with RMSE saved to: {plot_path}")

    plt.show()