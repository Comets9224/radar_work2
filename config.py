# config.py
import numpy as np
import os

# --- 核心参数 ---
DT = 0.1  # 统一的时间步长 (s)
TOTAL_SIMULATION_TIME = 20 # (s)

# --- 场景与目标配置 ---
OBSERVER_INITIAL_STATE = np.array([[0], [0], [10], [0]])  # [px, py, vx, vy]
TARGETS_CONFIG = [
    {'initial_state': np.array([[20], [5], [1], [0.5]]), 'model': 'CV'},
    {'initial_state': np.array([[35], [-15], [-0.5], [0.2], [-0.1], [0.05]]), 'model': 'CA'},
    {'initial_state': np.array([[50], [15], [1.5], [0.1]]), 'model': 'CV'},
    {'initial_state': np.array([[65], [-10], [1.0], [0.1]]), 'model': 'CV'},
    {'initial_state': np.array([[80], [-8], [3.0], [-0.5]]), 'model': 'CV'}
]
TARGET_COLORS = ['#FF0000', '#00FF00', '#00FFFF', '#FFFF00', '#FFA500', '#FF00FF']

# --- 运动模型 ---
def get_model_functions(model_type):
    if model_type == 'CV':
        q_std = 0.5
        F = lambda dt: np.array([[1, 0, dt, 0], [0, 1, 0, dt], [0, 0, 1, 0], [0, 0, 0, 1]])
        Q = lambda dt: np.array([[dt**4/4, 0, dt**3/2, 0], [0, dt**4/4, 0, dt**3/2], [dt**3/2, 0, dt**2, 0], [0, dt**3/2, 0, dt**2]]) * q_std**2
        return F, Q
    elif model_type == 'CA':
        q_std = 0.1
        F = lambda dt: np.array([[1, 0, dt, 0, 0.5*dt**2, 0], [0, 1, 0, dt, 0, 0.5*dt**2], [0, 0, 1, 0, dt, 0], [0, 0, 0, 1, 0, dt], [0, 0, 0, 0, 1, 0], [0, 0, 0, 0, 0, 1]])
        Q = lambda dt: np.eye(6) * q_std**2
        return F, Q
    raise ValueError(f"Unknown model type: {model_type}")

# --- 雷达参数 ---
RADAR_MAX_RANGE = 70.0
RADAR_MIN_RANGE = 2.0
RADAR_FOV_DEG = 90.0
PROB_DETECTION = 1.0
CLUTTER_RATE = 3
SIGMA_RANGE = 0.01
SIGMA_AZIMUTH_DEG = 0.01
SIGMA_VR = 0.2
R_MEASUREMENT = np.diag([SIGMA_RANGE**2, np.deg2rad(SIGMA_AZIMUTH_DEG)**2, SIGMA_VR**2])

# --- 新增：雷达分辨率参数 (用于严格P_fa计算) ---
RADAR_RANGE_RESOLUTION_M = 1.0
RADAR_AZIMUTH_RESOLUTION_DEG = 2.0

# --- 检测器参数 (DBSCAN) ---
DBSCAN_EPS = 4.5 # 这个值是默认值，会被ROC曲线生成过程临时修改
DBSCAN_MIN_SAMPLES = 1

# --- 新增：ROC曲线参数 ---
ROC_EPS_VALUES = [1.0, 1.5, 2.0, 2.7, 3.5, 4.5, 6.0, 8.0, 10.0, 12.0, 15.0] # 确保有多个值
# --- 跟踪器参数 ---
GATING_THRESHOLD_EUCLIDEAN = 5.0
M_CONFIRM = 3
MAX_CONSECUTIVE_MISSES = 15
N_CONFIRM_AGE = 20

# --- 输出路径 ---
OUTPUT_DIR = "output"
OUTPUT_DATA_DIR = os.path.join(OUTPUT_DIR, "data")
OUTPUT_PLOTS_DIR = os.path.join(OUTPUT_DIR, "plots")

def print_config_summary():
    print("--- Configuration Loaded ---")
    print(f"  DBSCAN EPS (default for main sim): {DBSCAN_EPS}")
    # ... (可以添加更多配置信息的打印)

if __name__ == '__main__':
    print_config_summary()
