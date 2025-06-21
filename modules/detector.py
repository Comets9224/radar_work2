# modules/detector.py
import numpy as np
from sklearn.cluster import DBSCAN


# import config as cfg # config will be passed as an argument

def detect_targets_by_clustering(measurements_at_t_polar, observer_state_at_t, config_obj):
    """
    使用DBSCAN对单帧雷达极坐标观测数据进行聚类检测。
    观测数据中的角度是相对于雷达朝向的。

    :param measurements_at_t_polar: 当前时刻的观测点列表 [[r1, theta_local1, vr1], ...] (相对于雷达)
    :param observer_state_at_t: 当前时刻观测者的状态 [px_obs, py_obs, vx_obs, vy_obs] (全局坐标)
    :param config_obj: 包含DBSCAN参数和RADAR_MIN_RANGE的配置对象
    :return: 检测到的目标信息列表，每个元素是一个字典:
             [{'position_global': [px_g, py_g], 'measurements_polar': [[r, theta, vr], ...]}, ...]
             其中 'measurements_polar' 包含了属于该聚类的原始极坐标测量值
    """
    if not measurements_at_t_polar:
        return []

    # 过滤掉小于最小探测距离的测量点
    filtered_measurements_polar = []
    for r, theta, vr in measurements_at_t_polar:
        if r >= config_obj.RADAR_MIN_RANGE:
            filtered_measurements_polar.append([r, theta, vr])

    if not filtered_measurements_polar:
        return []

    obs_x_g, obs_y_g, obs_vx_g, obs_vy_g = observer_state_at_t[:4]
    if not (np.isclose(obs_vx_g, 0) and np.isclose(obs_vy_g, 0)):
        observer_heading_rad = np.arctan2(obs_vy_g, obs_vx_g)
    else:
        observer_heading_rad = 0.0

    # 将过滤后的极坐标测量转换为全局笛卡尔坐标
    global_cartesian_points_for_clustering = []
    for r, theta_local, _ in filtered_measurements_polar:
        x_radar = r * np.cos(theta_local)  # 在雷达坐标系下的x
        y_radar = r * np.sin(theta_local)  # 在雷达坐标系下的y

        # 旋转到全局坐标系的相对位置
        x_global_rel = x_radar * np.cos(observer_heading_rad) - y_radar * np.sin(observer_heading_rad)
        y_global_rel = x_radar * np.sin(observer_heading_rad) + y_radar * np.cos(observer_heading_rad)

        # 加上观测者全局位置得到目标的全局位置
        global_x = obs_x_g + x_global_rel
        global_y = obs_y_g + y_global_rel
        global_cartesian_points_for_clustering.append([global_x, global_y])

    if not global_cartesian_points_for_clustering:
        return []

    global_cartesian_points_np = np.array(global_cartesian_points_for_clustering)

    # 执行DBSCAN聚类
    db = DBSCAN(eps=config_obj.DBSCAN_EPS, min_samples=config_obj.DBSCAN_MIN_SAMPLES).fit(global_cartesian_points_np)
    labels = db.labels_

    detected_targets_info_list = []
    unique_labels = set(labels)

    for label in unique_labels:
        if label != -1:  # -1表示DBSCAN中的噪声点，我们忽略它们
            cluster_indices = np.where(labels == label)[0]

            # 获取该聚类的所有全局笛卡尔坐标点
            cluster_points_global_cartesian = global_cartesian_points_np[cluster_indices]

            # 计算聚类中心作为检测到的目标位置 (全局坐标)
            target_position_global = np.mean(cluster_points_global_cartesian, axis=0)

            # 收集属于该聚类的原始极坐标测量值
            # filtered_measurements_polar 和 global_cartesian_points_np 是一一对应的
            original_polar_measurements_for_this_cluster = [filtered_measurements_polar[i] for i in cluster_indices]

            detected_targets_info_list.append({
                'position_global': target_position_global.tolist(),  # [px_g, py_g]
                'measurements_polar': original_polar_measurements_for_this_cluster  # list of [r, theta, vr]
            })

    return detected_targets_info_list