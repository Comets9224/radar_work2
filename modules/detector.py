# modules/detector.py
import numpy as np
from sklearn.cluster import DBSCAN
# import config as cfg # config 会通过参数传入

def detect_targets_by_clustering(measurements_at_t, observer_state_at_t, config):
    """
    使用DBSCAN对单帧雷达极坐标观测数据进行聚类检测。
    观测数据中的角度是相对于雷达朝向的。

    :param measurements_at_t: 当前时刻的观测点列表 [[r1, theta_local1, vr1], ...] (相对于雷达)
    :param observer_state_at_t: 当前时刻观测者的状态 [px_obs, py_obs, vx_obs, vy_obs] (全局坐标)
    :param config: 包含DBSCAN参数和RADAR_MIN_RANGE的配置对象
    :return: 检测到的目标在全局坐标系下的位置列表 [[px_g, py_g], ...]
    """
    if not measurements_at_t:
        return []

    # --- 新增的过滤步骤：根据最小探测距离过滤 ---
    filtered_measurements = []
    for r, theta, vr in measurements_at_t:
        if r >= config.RADAR_MIN_RANGE: # 只保留大于等于最小探测距离的点
            filtered_measurements.append([r, theta, vr])

    if not filtered_measurements: # 如果过滤后没有点了，直接返回
        return []
    # --- 过滤结束 ---

    obs_x_g, obs_y_g, obs_vx_g, obs_vy_g = observer_state_at_t
    observer_heading_rad = np.arctan2(obs_vy_g, obs_vx_g) if not (np.isclose(obs_vx_g, 0) and np.isclose(obs_vy_g, 0)) else 0

    global_cartesian_points = []
    # 使用过滤后的 filtered_measurements
    for r, theta_local, _ in filtered_measurements:
        x_radar = r * np.cos(theta_local)
        y_radar = r * np.sin(theta_local)
        x_global_rel = x_radar * np.cos(observer_heading_rad) - y_radar * np.sin(observer_heading_rad)
        y_global_rel = x_radar * np.sin(observer_heading_rad) + y_radar * np.cos(observer_heading_rad)
        global_x = obs_x_g + x_global_rel
        global_y = obs_y_g + y_global_rel
        global_cartesian_points.append([global_x, global_y])

    if not global_cartesian_points:
        return []

    global_cartesian_points_np = np.array(global_cartesian_points)

    db = DBSCAN(eps=config.DBSCAN_EPS, min_samples=config.DBSCAN_MIN_SAMPLES).fit(global_cartesian_points_np)
    labels = db.labels_

    detected_targets_global = []
    unique_labels = set(labels)
    for label in unique_labels:
        if label != -1:
            cluster_points_global = global_cartesian_points_np[labels == label]
            target_position_global = np.mean(cluster_points_global, axis=0)
            detected_targets_global.append(target_position_global.tolist())

    return detected_targets_global
