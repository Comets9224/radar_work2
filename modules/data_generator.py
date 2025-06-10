# modules/data_generator.py
import numpy as np
import config as cfg # 确保导入config
import math
import os

# ... (get_cv_matrices, get_ca_matrices, generate_true_trajectory, generate_observer_trajectory, transform_to_radar_polar_coords, generate_measurements_for_step, add_measurement_noise, apply_detection_probability 保持不变) ...
def get_cv_matrices(dt):
    F = np.array([[1, 0, dt, 0],
                  [0, 1, 0, dt],
                  [0, 0, 1, 0],
                  [0, 0, 0, 1]])
    Q = np.array([
        [dt**4/4, 0,       dt**3/2, 0      ],
        [0,       dt**4/4, 0,       dt**3/2],
        [dt**3/2, 0,       dt**2,   0      ],
        [0,       dt**3/2, 0,       dt**2  ]
    ]) * cfg.q_cv_accel_noise_std**2
    return F, Q

def get_ca_matrices(dt):
    F = np.array([[1, 0, dt, 0, 0.5*dt**2, 0],
                  [0, 1, 0, dt, 0, 0.5*dt**2],
                  [0, 0, 1, 0, dt, 0],
                  [0, 0, 0, 1, 0, dt],
                  [0, 0, 0, 0, 1, 0],
                  [0, 0, 0, 0, 0, 1]])
    Q = np.array([
        [cfg.DT**6/36, 0,            cfg.DT**5/12, 0,            cfg.DT**4/6, 0           ],
        [0,            cfg.DT**6/36, 0,            cfg.DT**5/12, 0,           cfg.DT**4/6 ],
        [cfg.DT**5/12, 0,            cfg.DT**4/4,  0,            cfg.DT**3/2, 0           ],
        [0,            cfg.DT**5/12, 0,            cfg.DT**4/4,  0,           cfg.DT**3/2 ],
        [cfg.DT**4/6,  0,            cfg.DT**3/2,  0,            cfg.DT**2,   0           ],
        [0,            cfg.DT**4/6,  0,            cfg.DT**3/2,  0,           cfg.DT**2   ]
    ]) * cfg.q_ca_jerk_noise_std**2
    return F, Q

def generate_true_trajectory(initial_state_pv, motion_model_type, total_time, dt, initial_state_full_ca=None):
    trajectory = []
    num_steps = int(total_time / dt)
    if motion_model_type == 'CV':
        current_state_cv = initial_state_pv.copy()
        trajectory.append(current_state_cv[:4])
        F_cv, _ = get_cv_matrices(dt)
        for _ in range(num_steps - 1):
            current_state_cv = F_cv @ current_state_cv
            trajectory.append(current_state_cv[:4].copy())
    elif motion_model_type == 'CA':
        if initial_state_full_ca is None:
            current_state_ca = np.array([initial_state_pv[0], initial_state_pv[1],
                                         initial_state_pv[2], initial_state_pv[3],
                                         0.0, 0.0])
        else:
            current_state_ca = initial_state_full_ca.copy()
        trajectory.append(current_state_ca[:4])
        F_ca, _ = get_ca_matrices(dt)
        for _ in range(num_steps - 1):
            current_state_ca = F_ca @ current_state_ca
            trajectory.append(current_state_ca[:4].copy())
    else:
        raise ValueError(f"Unknown motion model: {motion_model_type}")
    return np.array(trajectory)

def generate_observer_trajectory(initial_state, total_time, dt):
    trajectory = [initial_state.copy()]
    current_state = initial_state.copy()
    F, _ = get_cv_matrices(dt)
    num_steps = int(total_time / dt)
    for _ in range(num_steps - 1):
        current_state = F @ current_state
        trajectory.append(current_state.copy())
    return np.array(trajectory)

def transform_to_radar_polar_coords(target_global_state, observer_global_state):
    dx_global = target_global_state[0] - observer_global_state[0]
    dy_global = target_global_state[1] - observer_global_state[1]
    dvx_global = target_global_state[2] - observer_global_state[2]
    dvy_global = target_global_state[3] - observer_global_state[3]
    observer_heading_rad = np.arctan2(observer_global_state[3], observer_global_state[2]) if not (np.isclose(observer_global_state[2], 0) and np.isclose(observer_global_state[3], 0)) else 0
    x_radar_rel = dx_global * np.cos(observer_heading_rad) + dy_global * np.sin(observer_heading_rad)
    y_radar_rel = -dx_global * np.sin(observer_heading_rad) + dy_global * np.cos(observer_heading_rad)
    dist = np.sqrt(x_radar_rel**2 + y_radar_rel**2)
    azimuth_local_rad = np.arctan2(y_radar_rel, x_radar_rel)
    if dist < 1e-6:
        radial_velocity = 0.0
    else:
        radial_velocity = (dx_global * dvx_global + dy_global * dvy_global) / dist
    return [dist, azimuth_local_rad, radial_velocity]

def generate_measurements_for_step(true_target_states_at_step, observer_state_at_step, config):
    ideal_measurements = []
    for target_state in true_target_states_at_step:
        measurement = transform_to_radar_polar_coords(target_state, observer_state_at_step)
        dist, azimuth_local_rad, _ = measurement
        # 应用最小和最大探测距离，以及FOV
        if config.RADAR_MIN_RANGE <= dist <= config.RADAR_MAX_RANGE and \
           abs(azimuth_local_rad) <= config.RADAR_FOV_RAD / 2:
            ideal_measurements.append(measurement)
    return ideal_measurements

def add_measurement_noise(ideal_measurements, R_matrix):
    noisy_measurements = []
    for meas in ideal_measurements:
        noise = np.random.multivariate_normal(np.zeros(3), R_matrix)
        noisy_meas = np.array(meas) + noise
        noisy_meas[1] = (noisy_meas[1] + np.pi) % (2 * np.pi) - np.pi
        noisy_measurements.append(noisy_meas.tolist())
    return noisy_measurements

def apply_detection_probability(true_measurements_with_noise, prob_detection):
    detected_measurements = []
    for measurement in true_measurements_with_noise:
        if np.random.rand() < prob_detection:
            detected_measurements.append(measurement)
    return detected_measurements

def generate_false_alarms(clutter_rate, min_range, max_range, fov_rad): # 添加 min_range
    false_alarms = []
    num_clutter = np.random.poisson(clutter_rate)
    for _ in range(num_clutter):
        dist = np.random.uniform(min_range, max_range) # 在最小和最大距离之间生成
        angle_rad = np.random.uniform(-fov_rad / 2, fov_rad / 2)
        rad_vel = (np.random.rand() - 0.5) * 20
        false_alarms.append([dist, angle_rad, rad_vel])
    return false_alarms

def generate_full_dataset(config):
    print("  Generating observer trajectory...")
    observer_trajectory = generate_observer_trajectory(
        config.OBSERVER_INITIAL_STATE,
        config.TOTAL_SIMULATION_TIME,
        config.TIME_STEP
    )
    true_target_trajectories = []
    print("  Generating true target trajectories...")
    for i in range(config.NUM_TARGETS):
        initial_state_pv = config.TARGET_INITIAL_STATES[i]
        motion_model = config.TARGET_MOTION_MODELS[i]
        initial_state_full_ca = None
        if motion_model == 'CA':
            initial_state_full_ca = config.TARGET_CA_INITIAL_STATES_FULL.get(i)
        traj = generate_true_trajectory(
            initial_state_pv, motion_model, config.TOTAL_SIMULATION_TIME,
            config.TIME_STEP, initial_state_full_ca=initial_state_full_ca
        )
        true_target_trajectories.append(traj)

    num_steps = len(observer_trajectory)
    all_final_measurements = []
    print("  Generating simulated radar measurements per step...")
    for k in range(num_steps):
        observer_state_k = observer_trajectory[k]
        true_target_states_k = [traj[k] for traj in true_target_trajectories]
        ideal_measurements_k = generate_measurements_for_step(
            true_target_states_k, observer_state_k, config
        )
        noisy_true_measurements_k = add_measurement_noise(
            ideal_measurements_k, config.R_MEASUREMENT
        )
        detected_targets_k = apply_detection_probability(
            noisy_true_measurements_k, config.PROB_DETECTION
        )
        clutter_points_k = generate_false_alarms( # 传递min_range
            config.CLUTTER_RATE,
            config.RADAR_MIN_RANGE,
            config.RADAR_MAX_RANGE,
            config.RADAR_FOV_RAD
        )
        combined_measurements_k = detected_targets_k + clutter_points_k
        np.random.shuffle(combined_measurements_k)
        all_final_measurements.append(combined_measurements_k)
        if (k + 1) % (num_steps // 10 or 1) == 0:
             print(f"    Generated measurements for step {k+1}/{num_steps}")

    os.makedirs(cfg.OUTPUT_DATA_DIR, exist_ok=True)
    np.save(os.path.join(cfg.OUTPUT_DATA_DIR, 'true_observer_trajectory.npy'), observer_trajectory)
    for i, traj in enumerate(true_target_trajectories):
        np.save(os.path.join(cfg.OUTPUT_DATA_DIR, f'true_target_{i+1}_trajectory.npy'), traj)
    import pickle
    with open(os.path.join(cfg.OUTPUT_DATA_DIR, 'radar_measurements.pkl'), 'wb') as f:
        pickle.dump(all_final_measurements, f)
    print(f"  All data generated and saved to {cfg.OUTPUT_DATA_DIR}")
    return true_target_trajectories, observer_trajectory, all_final_measurements

def print_motion_equations():
    # (保持不变)
    dt_symbol = 'Δt'
    print("\n--- Kinematic Equations ---")
    print("Global Coordinate System: X-axis forward, Y-axis left")
    print("\nObserver (Ego Vehicle) - Constant Velocity (CV):")
    print(f"  px_k = px_(k-1) + vx_(k-1) * {dt_symbol}")
    print(f"  py_k = py_(k-1) + vy_(k-1) * {dt_symbol}")
    print(f"  vx_k = vx_(k-1)")
    print(f"  vy_k = vy_(k-1)")
    print(f"  State vector x_k = [px_k, py_k, vx_k, vy_k]^T")
    F_cv_example, Q_cv_example = get_cv_matrices(1)
    print(f"  State Transition Matrix F_cv (example dt=1s) = \n{F_cv_example}")
    print(f"  Process Noise Covariance Q_CV (example dt=1s, q_accel_std={cfg.q_cv_accel_noise_std}) = \n{Q_cv_example}")
    print("\nTarget - Constant Velocity (CV): (Same as Observer for F, Q might differ if configured)")
    print("\nTarget - Constant Acceleration (CA):")
    print(f"  px_k = px_(k-1) + vx_(k-1)*{dt_symbol} + 0.5*ax_(k-1)*{dt_symbol}^2")
    print(f"  py_k = py_(k-1) + vy_(k-1)*{dt_symbol} + 0.5*ay_(k-1)*{dt_symbol}^2")
    print(f"  vx_k = vx_(k-1) + ax_(k-1)*{dt_symbol}")
    print(f"  vy_k = vy_(k-1) + ay_(k-1)*{dt_symbol}")
    print(f"  ax_k = ax_(k-1)")
    print(f"  ay_k = ay_(k-1)")
    print(f"  State vector x_k = [px_k, py_k, vx_k, vy_k, ax_k, ay_k]^T")
    F_ca_example, Q_ca_example = get_ca_matrices(1)
    print(f"  State Transition Matrix F_ca (example dt=1s) = \n{F_ca_example}")
    print(f"  Process Noise Covariance Q_CA (example dt=1s, q_jerk_std={cfg.q_ca_jerk_noise_std}) = \n{Q_ca_example}")
    print("--------------------")

if __name__ == '__main__':
    print("Testing data_generator.py...")
    true_targets, obs_traj, measurements = generate_full_dataset(cfg)
    print(f"\nGenerated {len(measurements)} frames of radar data.")
    if measurements and measurements[0]:
        print(f"First frame measurements ({len(measurements[0])} points): {measurements[0][:3]}")
    elif measurements:
        print("First frame has 0 measurements.")
    print(f"Observer trajectory shape: {obs_traj.shape}")
    print(f"Target 1 trajectory shape: {true_targets[0].shape}")
