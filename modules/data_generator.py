# modules/data_generator.py
import numpy as np
import os
import config as cfg


def generate_true_trajectories(config):
    """根据config中的设置，生成所有目标的真实轨迹"""
    all_trajectories = []
    num_steps = int(config.TOTAL_SIMULATION_TIME / config.DT)

    for target_cfg in config.TARGETS_CONFIG:
        model_type = target_cfg['model']
        initial_state = target_cfg['initial_state']
        F_func, _ = config.get_model_functions(model_type)
        F = F_func(config.DT)

        current_state = initial_state.copy().reshape(-1, 1)
        trajectory = [current_state.flatten()]

        for _ in range(num_steps - 1):
            current_state = F @ current_state
            trajectory.append(current_state.flatten())

        all_trajectories.append(np.array(trajectory))
    return all_trajectories


def generate_observer_trajectory(config):
    """生成观测者的轨迹"""
    num_steps = int(config.TOTAL_SIMULATION_TIME / config.DT)
    F_func, _ = config.get_model_functions('CV')
    F = F_func(config.DT)

    current_state = config.OBSERVER_INITIAL_STATE.copy()
    trajectory = [current_state.flatten()]

    for _ in range(num_steps - 1):
        current_state = F @ current_state
        trajectory.append(current_state.flatten())

    return np.array(trajectory)


def transform_to_radar_polar_coords(target_state, observer_state):
    """将目标的全局状态转换为相对于观测者的局部极坐标"""
    px, py, vx, vy = target_state[:4]
    obs_px, obs_py, obs_vx, obs_vy = observer_state

    delta_px = px - obs_px
    delta_py = py - obs_py

    dist = np.sqrt(delta_px ** 2 + delta_py ** 2)
    angle_global = np.arctan2(delta_py, delta_px)
    observer_heading = np.arctan2(obs_vy, obs_vx)

    angle_local = angle_global - observer_heading
    angle_local = (angle_local + np.pi) % (2 * np.pi) - np.pi

    rad_vel = (vx * delta_px + vy * delta_py) / dist if dist > 0 else 0

    return [dist, angle_local, rad_vel]


def generate_measurements(true_trajectories, observer_trajectory, config):
    """生成带有噪声、漏检和范围/FOV限制的观测数据"""
    measurements_per_step = []
    num_steps = len(observer_trajectory)

    for k in range(num_steps):
        observer_state_k = observer_trajectory[k]
        step_measurements = []

        for traj in true_trajectories:
            if k < len(traj):
                target_state_k = traj[k]

                dist, angle_local, rad_vel = transform_to_radar_polar_coords(target_state_k, observer_state_k)

                if np.random.rand() < config.PROB_DETECTION and \
                        config.RADAR_MIN_RANGE <= dist <= config.RADAR_MAX_RANGE and \
                        abs(np.rad2deg(angle_local)) <= config.RADAR_FOV_DEG / 2:
                    noise = np.random.randn(3) * np.sqrt(np.diag(config.R_MEASUREMENT))
                    noisy_measurement = [dist, angle_local, rad_vel] + noise
                    step_measurements.append(noisy_measurement)

        measurements_per_step.append(step_measurements)
    return measurements_per_step


def generate_false_alarms(config):
    """在整个仿真时间和空间内生成虚警点"""
    num_steps = int(config.TOTAL_SIMULATION_TIME / config.DT)
    all_false_alarms = []

    for _ in range(num_steps):
        num_clutter = np.random.poisson(config.CLUTTER_RATE)
        step_clutter = []
        for _ in range(num_clutter):
            dist = config.RADAR_MIN_RANGE + np.random.rand() * (config.RADAR_MAX_RANGE - config.RADAR_MIN_RANGE)
            angle_local = np.deg2rad((np.random.rand() - 0.5) * config.RADAR_FOV_DEG)
            rad_vel = (np.random.rand() - 0.5) * 40
            step_clutter.append([dist, angle_local, rad_vel])
        all_false_alarms.append(step_clutter)
    return all_false_alarms


def generate_full_dataset(config):
    """生成包含所有元素的完整数据集"""
    observer_traj = generate_observer_trajectory(config)
    true_target_trajs = generate_true_trajectories(config)
    true_measurements = generate_measurements(true_target_trajs, observer_traj, config)
    false_alarms = generate_false_alarms(config)

    full_measurements_per_step = []
    for k in range(len(observer_traj)):
        combined_measurements = true_measurements[k] + false_alarms[k]
        np.random.shuffle(combined_measurements)
        full_measurements_per_step.append(combined_measurements)

    # --- 这是修正了数据保存错误的地方 ---
    data_path = os.path.join(config.OUTPUT_DATA_DIR, "simulation_data.npz")
    np.savez(data_path,
             true_trajs=np.array(true_target_trajs, dtype=object),
             obs_traj=observer_traj,
             measurements=np.array(full_measurements_per_step, dtype=object))
    # --- 修正结束 ---

    return true_target_trajs, observer_traj, full_measurements_per_step


def print_motion_equations():
    """打印运动学方程和示例矩阵"""
    dt_symbol = 'Δt'
    print("\n--- Kinematic Equations ---")

    F_cv_func, _ = cfg.get_model_functions('CV')
    F_cv_example = F_cv_func(1)
    print("\nObserver / Target - Constant Velocity (CV):")
    print(f"  State Transition Matrix F_cv (dt=1s):\n{F_cv_example}")

    F_ca_func, _ = cfg.get_model_functions('CA')
    F_ca_example = F_ca_func(1)
    print("\nTarget - Constant Acceleration (CA):")
    print(f"  State Transition Matrix F_ca (dt=1s):\n{F_ca_example}")
    print("--------------------")