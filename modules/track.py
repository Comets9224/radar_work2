# modules/track.py
import numpy as np
from .kalman_filter import ExtendedKalmanFilter
import config as cfg

class Track:
    _next_id = 1
    def __init__(self, initial_measurement, observer_state, dt):
        self.track_id = Track._next_id
        Track._next_id += 1
        self.age = 1
        self.hits = 1
        self.misses = 0
        self.state = 'Tentative'
        self.history = []

        r, theta_local, _ = initial_measurement
        obs_px, obs_py, obs_vx, obs_vy = observer_state.flatten()
        obs_heading = np.arctan2(obs_vy, obs_vx)
        x_rel = r * np.cos(theta_local)
        y_rel = r * np.sin(theta_local)
        px_g = obs_px + x_rel * np.cos(obs_heading) - y_rel * np.sin(obs_heading)
        py_g = obs_py + x_rel * np.sin(obs_heading) + y_rel * np.cos(obs_heading)
        x_init = np.array([[px_g], [py_g], [0.0], [0.0]])
        pos_var = cfg.R_MEASUREMENT[0, 0] * 4
        vel_var = 50**2
        P_init = np.diag([pos_var, pos_var, vel_var, vel_var])
        F_func, Q_func = cfg.get_model_functions('CV')
        F, Q = F_func(dt), Q_func(dt)
        self.kf = ExtendedKalmanFilter(x_init, P_init, F, Q, cfg.R_MEASUREMENT)
        self.history.append(self.kf.x.flatten().copy())

    def predict(self):
        self.age += 1
        self.kf.predict()

    def update(self, measurement, observer_state):
        if self.kf.update(measurement, observer_state):
            self.hits += 1
            self.misses = 0
        else:
            self.mark_missed()
        self.history.append(self.kf.x.flatten().copy())

    def mark_missed(self):
        self.misses += 1
        self.history.append(self.kf.x.flatten().copy())

    def get_predicted_position(self):
        return self.kf.x[:2].flatten()