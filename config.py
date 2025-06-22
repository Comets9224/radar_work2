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
    # 目标 2 (绿色) - 起点向下移动
    {'initial_state': np.array([[35], [-15], [-0.5], [0.2], [-0.1], [0.05]]), 'model': 'CA'},

    # 目标 3 (青色/蓝色) - 移动到轨迹另一侧（左侧）
    {'initial_state': np.array([[50], [15], [1.5], [0.1]]), 'model': 'CV'},

    # 目标 4 (黄色) - 不变
    {'initial_state': np.array([[65], [-10], [1.0], [0.1]]), 'model': 'CV'},

    # 目标 5 (橙色) - 提高速度以增长轨迹
    {'initial_state': np.array([[80], [-8], [3.0], [-0.5]]), 'model': 'CV'}
]
TARGET_COLORS = ['#FF0000', '#00FF00', '#00FFFF', '#FFFF00', '#FFA500', '#FF00FF'] # Red, Green, Cyan, etc.

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
        Q = lambda dt: np.eye(6) * q_std**2 # 简化Q矩阵，实际CA的Q更复杂，但对角阵常用
        return F, Q
    raise ValueError(f"Unknown model type: {model_type}")

# --- 雷达参数 ---
RADAR_MAX_RANGE = 70.0
RADAR_MIN_RANGE = 2.0
RADAR_FOV_DEG = 90.0
PROB_DETECTION = 0.9  # 理想检测概率，在generate_measurements中使用
CLUTTER_RATE = 5      # 平均每帧在FOV内产生的杂波点数 (泊松分布均值)
SIGMA_RANGE = 0.5     # 距离量测标准差 (m)
SIGMA_AZIMUTH_DEG = 1.0 # 方位角量测标准差 (度)
SIGMA_VR = 0.2        # 径向速度量测标准差 (m/s)
R_MEASUREMENT = np.diag([SIGMA_RANGE**2, np.deg2rad(SIGMA_AZIMUTH_DEG)**2, SIGMA_VR**2]) # 量测噪声协方差矩阵

# --- 新增：雷达分辨率参数 (用于严格P_fa计算) ---
RADAR_RANGE_RESOLUTION_M = 1.0  # 示例值：距离分辨率1米
RADAR_AZIMUTH_RESOLUTION_DEG = 2.0 # 示例值：方位角分辨率2度
# 注意：这些分辨率值应该与你的雷达仿真场景和期望分析的P_fa水平相匹配。
# 如果分辨率设置得过细（值很小），N_fa_opp会非常大，导致计算出的P_fa非常小。
# 如果分辨率设置得过粗（值很大），N_fa_opp会很小，P_fa可能会偏高。

# --- 检测器参数 (DBSCAN) ---
DBSCAN_EPS = 2.7  # DBSCAN的邻域半径 (米)
DBSCAN_MIN_SAMPLES = 2 # DBSCAN形成簇的最小样本数

# --- 跟踪器参数 ---
GATING_THRESHOLD_EUCLIDEAN = 5.0  # 航迹与检测关联的欧氏距离门限
M_CONFIRM = 1          # 暂定航迹转为确认航迹所需的连续命中次数
MAX_CONSECUTIVE_MISSES = 10 # 确认航迹在被删除前允许的最大连续丢失次数
N_CONFIRM_AGE = 10     # 暂定航迹在未达到M_CONFIRM时，能存活的最大帧数（年龄）

# --- 输出路径 ---
OUTPUT_DIR = "output"
OUTPUT_DATA_DIR = os.path.join(OUTPUT_DIR, "data")
OUTPUT_PLOTS_DIR = os.path.join(OUTPUT_DIR, "plots")

def print_config_summary():
    """打印关键配置信息摘要"""
    print("--- Configuration Loaded ---")
    print(f"  Simulation Time: {TOTAL_SIMULATION_TIME}s, DT: {DT}s")
    print(f"  Radar Max Range: {RADAR_MAX_RANGE}m, FOV: {RADAR_FOV_DEG}deg")
    print(f"  Radar Range Resolution: {RADAR_RANGE_RESOLUTION_M}m, Azimuth Resolution: {RADAR_AZIMUTH_RESOLUTION_DEG}deg (for P_fa calc)")
    print(f"  DBSCAN EPS: {DBSCAN_EPS}, Min Samples: {DBSCAN_MIN_SAMPLES}")
    print(f"  Clutter Rate: {CLUTTER_RATE} points/frame (Poisson mean)")
    print(f"  Prob. of Detection (ideal): {PROB_DETECTION}")
    print("--------------------------")

# 可以在这里添加一个主执行块来测试打印配置（如果单独运行此文件）
if __name__ == '__main__':
    print_config_summary()