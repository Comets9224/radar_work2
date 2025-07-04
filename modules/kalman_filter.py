# modules/kalman_filter.py
import numpy as np

class ExtendedKalmanFilter:
    def __init__(self, x_init, P_init, F, Q, R):
        self.x = x_init
        self.P = P_init
        self.F = F
        self.Q = Q
        self.R = R
        self.I = np.eye(x_init.shape[0])

    def predict(self):
        self.x = self.F @ self.x
        self.P = self.F @ self.P @ self.F.T + self.Q

    def hx(self, x_state, observer_state):
        px, py, vx, vy = x_state.flatten()
        obs_px, obs_py, obs_vx, obs_vy = observer_state.flatten()
        delta_px = px - obs_px
        delta_py = py - obs_py
        range_val = np.sqrt(delta_px**2 + delta_py**2)
        if range_val < 0.1: return None
        angle_global = np.arctan2(delta_py, delta_px)
        observer_heading = np.arctan2(obs_vy, obs_vx)
        predicted_theta_local = angle_global - observer_heading
        predicted_theta_local = (predicted_theta_local + np.pi) % (2 * np.pi) - np.pi
        predicted_vr = (vx * delta_px + vy * delta_py) / range_val if range_val > 0 else 0
        return np.array([[range_val], [predicted_theta_local], [predicted_vr]])

    def jacobian_H(self, x_state, observer_state):
        px, py, vx, vy = x_state.flatten()
        obs_px, obs_py, obs_vx, obs_vy = observer_state.flatten()
        delta_px = px - obs_px
        delta_py = py - obs_py
        range_sq = delta_px**2 + delta_py**2
        range_val = np.sqrt(range_sq)
        if range_val < 0.1: return None
        H = np.zeros((3, 4))
        H[0, 0] = delta_px / range_val
        H[0, 1] = delta_py / range_val
        H[1, 0] = -delta_py / range_sq
        H[1, 1] = delta_px / range_sq
        vr_predicted = (vx * delta_px + vy * delta_py) / range_val if range_val > 0 else 0
        H[2, 0] = (vx - obs_vx) / range_val - (vr_predicted * delta_px) / range_sq
        H[2, 1] = (vy - obs_vy) / range_val - (vr_predicted * delta_py) / range_sq
        H[2, 2] = delta_px / range_val
        H[2, 3] = delta_py / range_val
        return H

    def update(self, z_measurement, observer_state):
        z_pred = self.hx(self.x, observer_state)
        if z_pred is None: return False
        H = self.jacobian_H(self.x, observer_state)
        if H is None: return False
        y = np.array(z_measurement).reshape(3, 1) - z_pred
        if y[1, 0] > np.pi: y[1, 0] -= 2 * np.pi
        if y[1, 0] < -np.pi: y[1, 0] += 2 * np.pi
        S = H @ self.P @ H.T + self.R
        try:
            K = self.P @ H.T @ np.linalg.inv(S)
        except np.linalg.LinAlgError:
            return False
        self.x = self.x + K @ y
        I_KH = self.I - K @ H
        self.P = I_KH @ self.P @ I_KH.T + K @ self.R @ K.T
        return True