# modules/visualizer.py
import matplotlib.pyplot as plt
import numpy as np
import os
import config as cfg

def plot_results(observer_trajectory, true_target_trajectories, measurements_over_time, detections_over_time, tracks_over_time):
    """
    绘制所有仿真结果的最终图表。
    """
    fig, ax = plt.subplots(figsize=(16, 12))
    ax.set_facecolor('black')
    fig.patch.set_facecolor('black')
    ax.spines['bottom'].set_color('gray')
    ax.spines['top'].set_color('gray')
    ax.spines['right'].set_color('gray')
    ax.spines['left'].set_color('gray')
    ax.tick_params(axis='x', colors='gray')
    ax.tick_params(axis='y', colors='gray')

    ax.set_xlabel("X (m) - Global Coordinate System", color='white', fontsize=12)
    ax.set_ylabel("Y (m) - Global Coordinate System", color='white', fontsize=12)
    ax.set_title("Radar Simulation: Trajectories, Detections, and Tracks", color='white', fontsize=16)
    ax.grid(True, linestyle='--', alpha=0.2, color='gray')
    ax.axis('equal')

    # 1. 绘制原始雷达观测点 (灰色x)
    all_meas_x, all_meas_y = [], []
    for k, step_meas in enumerate(measurements_over_time):
        obs_state_k = observer_trajectory[k]
        obs_x, obs_y, obs_vx, obs_vy = obs_state_k
        obs_heading = np.arctan2(obs_vy, obs_vx)
        for r, theta_local, _ in step_meas:
            x_rel = r * np.cos(theta_local)
            y_rel = r * np.sin(theta_local)
            px_g = obs_x + x_rel * np.cos(obs_heading) - y_rel * np.sin(obs_heading)
            py_g = obs_y + x_rel * np.sin(obs_heading) + y_rel * np.cos(obs_heading)
            all_meas_x.append(px_g)
            all_meas_y.append(py_g)
    ax.scatter(all_meas_x, all_meas_y, color='dimgray', marker='x', s=15, alpha=0.4, label='Raw Measurements (incl. Clutter)')

    # 2. 绘制DBSCAN检测结果 (紫色空心圆圈)
    all_det_x, all_det_y = [], []
    for step_dets in detections_over_time:
        for det in step_dets:
            all_det_x.append(det['position_global'][0])
            all_det_y.append(det['position_global'][1])
    ax.scatter(all_det_x, all_det_y, s=40, facecolors='none', edgecolors='magenta', alpha=0.9, linewidths=1.5, label='DBSCAN Detections')

    # 3. 绘制观测者轨迹 (蓝色实线)
    ax.plot(observer_trajectory[:, 0], observer_trajectory[:, 1], color='cyan', linestyle='-', label='Observer Trajectory', lw=2)
    ax.plot(observer_trajectory[0, 0], observer_trajectory[0, 1], marker='s', color='cyan', markersize=8, label='Observer Start')

    # 4. 绘制真实目标轨迹 (彩色实线)
    for i, traj in enumerate(true_target_trajectories):
        color = cfg.TARGET_COLORS[i % len(cfg.TARGET_COLORS)]
        ax.plot(traj[:, 0], traj[:, 1], color=color, linestyle='-', label=f'True Target {i+1}', lw=2.5)
        ax.plot(traj[0, 0], traj[0, 1], marker='o', color=color, markersize=8, label=f'Target {i+1} Start')

    # 5. 绘制最终的跟踪轨迹 (带ID的彩色虚线)
    plotted_ids = set()
    for track_info in tracks_over_time:
        track_id = track_info['id']
        history = np.array(track_info['history'])
        state = track_info['state']
        # 为跟踪轨迹使用与真实轨迹相同的颜色，但样式不同
        color = cfg.TARGET_COLORS[(track_id - 1) % len(cfg.TARGET_COLORS)]
        if history.shape[0] > 1 and state == 'Confirmed':
            label = f'Est. Track {track_id}' if track_id not in plotted_ids else "_nolegend_"
            ax.plot(history[:, 0], history[:, 1], color=color, linestyle='--', marker='.', markersize=4, lw=2.0, label=label)
            ax.text(history[-1, 0] + 1, history[-1, 1] + 1, f'ID:{track_id}', color=color, fontsize=10, weight='bold')
            plotted_ids.add(track_id)

    # 最终绘图调整
    legend = ax.legend(facecolor='black', framealpha=0.8, edgecolor='gray', loc='upper left')
    for text in legend.get_texts():
        text.set_color('white')

    plot_path = os.path.join(cfg.OUTPUT_PLOTS_DIR, "final_tracking_result.png")
    plt.savefig(plot_path, facecolor='black', dpi=200, bbox_inches='tight')
    print(f"\nFinal plot saved to: {plot_path}")