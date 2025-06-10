# modules/track.py
import numpy as np
from .kalman_filter import ExtendedKalmanFilter
import config as cfg
# from . import data_generator # 不再需要，因为F,Q函数通过cfg.get_model_functions获取

class Track:
    _next_id = 1

    def __init__(self, initial_measurement_polar, observer_state_global, current_time, dt):
        """
        初始化一个航迹 (修正版，采用零速度初始化)
        :param initial_measurement_polar: 用于初始化航迹的第一个雷达观测 [r, theta_local, vr]
        :param observer_state_global: 当时观测者的状态 [px, py, vx, vy]
        :param current_time: 当前时间
        :param dt: 时间步长
        """
        self.track_id = Track._next_id
        Track._next_id += 1

        self.age = 1
        self.time_created = current_time
        self.time_updated = current_time
        self.hits = 1
        self.misses = 0
        self.state = 'Tentative'
        self.history = []

        # --- 核心修正：稳健的零速度初始化 ---

        # 1. 将第一个观测点从雷达局部极坐标转换为全局笛卡尔坐标
        r, theta_local, _ = initial_measurement_polar  # 我们只使用r和theta来确定初始位置
        obs_px, obs_py, obs_vx, obs_vy = observer_state_global.flatten() # 确保是一维数组

        # 计算观测者当前的航向角 (全局坐标系下，X轴正向为0度，逆时针为正)
        # 如果观测者速度接近0，则航向角意义不大，可以设为0或保持上一次的有效航向
        if not (np.isclose(obs_vx, 0) and np.isclose(obs_vy, 0)):
            observer_heading_rad = np.arctan2(obs_vy, obs_vx)
        else:
            observer_heading_rad = 0 # 或者使用一个默认/上一次的航向

        # 首先将(r, theta_local)转换为相对于车辆（雷达）的笛卡尔坐标 (x_rel_radar向前, y_rel_radar向左)
        x_rel_radar = r * np.cos(theta_local)
        y_rel_radar = r * np.sin(theta_local)

        # 然后根据车辆的航向和位置，旋转并平移到全局坐标系
        # 旋转矩阵: 从雷达坐标系到全局坐标系
        # x_global = x_radar * cos(heading) - y_radar * sin(heading)
        # y_global = x_radar * sin(heading) + y_radar * cos(heading)
        px_g = obs_px + x_rel_radar * np.cos(observer_heading_rad) - y_rel_radar * np.sin(observer_heading_rad)
        py_g = obs_py + x_rel_radar * np.sin(observer_heading_rad) + y_rel_radar * np.cos(observer_heading_rad)

        # 2. 初始化状态向量 x_init (速度设为0)
        vx_g_init = 0.0
        vy_g_init = 0.0
        x_init = np.array([[px_g], [py_g], [vx_g_init], [vy_g_init]])

        # 3. 初始化协方差矩阵 P_init (为速度设置一个非常大的不确定性)
        # 位置的不确定性可以从观测噪声R中得到一个大概的估计
        # R_MEASUREMENT[0,0] is sigma_range^2
        # 横向位置误差近似为 (r * sigma_azimuth)^2
        # 纵向位置误差近似为 sigma_range^2
        # 简单起见，可以用一个较大的固定值或基于R(0,0)
        pos_variance_factor = 5 # 放大因子，表示初始位置估计的不确定性比单次测量大
        var_px_init = pos_variance_factor * ( (cfg.SIGMA_RANGE * np.cos(theta_local))**2 + (r * cfg.SIGMA_AZIMUTH * np.sin(theta_local))**2 )
        var_py_init = pos_variance_factor * ( (cfg.SIGMA_RANGE * np.sin(theta_local))**2 + (r * cfg.SIGMA_AZIMUTH * np.cos(theta_local))**2 )
        # 确保方差不为零或过小
        var_px_init = max(var_px_init, (cfg.SIGMA_RANGE)**2)
        var_py_init = max(var_py_init, (0.5 * cfg.RADAR_MIN_RANGE * cfg.SIGMA_AZIMUTH)**2) # 最小横向不确定性


        # 速度的不确定性要设得非常大
        vel_variance = 50**2  # (m/s)^2, 速度标准差为50m/s

        P_init = np.diag([var_px_init, var_py_init, vel_variance, vel_variance])
        # --- 修正结束 ---

        # 创建EKF实例
        self.motion_model_type = 'CV'  # 新航迹默认使用最简单的CV模型
        
        # 从 config.py 获取模型矩阵的生成函数
        # cfg.get_model_matrices 返回 F 和 Q 矩阵
        # EKF 的 F_func 和 Q_func 期望是函数，所以用 lambda 包装
        F_func = lambda d_t_val: cfg.get_model_matrices(self.motion_model_type, d_t_val)[0]
        Q_func = lambda d_t_val: cfg.get_model_matrices(self.motion_model_type, d_t_val)[1]
        
        # 如果要支持CA模型初始化，需要确保EKF能处理6D状态
        if self.motion_model_type == 'CA': # 虽然当前默认是CV，但保留此分支的逻辑
            print(f"Warning: Track ID {self.track_id} - CA model selected for init, but EKF is 4D. Needs 6D EKF and state for full CA tracking.")
            # 如果要用CA，需要修改x_init和P_init为6维，并传递CA的F和Q
            # x_init = np.vstack((x_init, [[0], [0]])) # 添加ax, ay
            # P_init = np.diag(np.diag(P_init).tolist() + [cfg.q_ca_jerk_noise_std**2 * dt, cfg.q_ca_jerk_noise_std**2 * dt]) # 粗略的加速度初始不确定性
            # F_func = lambda d_t_val: cfg.get_model_matrices('CA', d_t_val)[0]
            # Q_func = lambda d_t_val: cfg.get_model_matrices('CA', d_t_val)[1]
            # 当前会回退到使用CV的F_func和Q_func，因为EKF是4D的

        self.kf = ExtendedKalmanFilter(x_init, P_init, F_func, Q_func, cfg.R_MEASUREMENT, dt)
        self.history.append(self.kf.x.flatten().copy())

    def predict(self):
        self.age += 1
        self.kf.predict()

    def update(self, measurement_polar, observer_state_global, current_time):
        update_successful = self.kf.update(measurement_polar, observer_state_global)
        if update_successful:
            self.hits += 1
            self.misses = 0
            self.time_updated = current_time
            self.history.append(self.kf.x.flatten().copy())
            return True
        else:
            self.mark_missed()
            return False

    def mark_missed(self):
        self.misses += 1
        self.history.append(self.kf.x.flatten().copy())

    def get_predicted_position(self):
        return self.kf.x[:2].flatten()

    def get_predicted_measurement_polar(self, observer_state_global):
        return self.kf.hx(self.kf.x, observer_state_global)

    def get_innovation_covariance(self, observer_state_global):
        H = self.kf.jacobian_H(self.kf.x, observer_state_global)
        if H is None or np.all(H == 0):
            return None
        S = H @ self.kf.P @ H.T + self.kf.R
        return S

    def get_current_estimate_global_cartesian(self):
        return self.kf.x.flatten()
