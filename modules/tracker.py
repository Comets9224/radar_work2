# modules/tracker.py
import numpy as np
from scipy.optimize import linear_sum_assignment
# from scipy.spatial.distance import cdist # 不再需要
from .track import Track
import config as cfg # 确保config能被导入
# from . import data_generator # tracker.py本身不直接调用data_generator的矩阵函数

class Tracker:
    def __init__(self):
        self.tracks = []
        self.dt = cfg.TIME_STEP
        Track._next_id = 1

    def _calculate_cost_matrix_mahalanobis(self, tracks, current_polar_measurements, observer_state_global):
        num_tracks = len(tracks)
        num_detections = len(current_polar_measurements)
        cost_matrix = np.full((num_tracks, num_detections), np.inf)

        if num_tracks == 0 or num_detections == 0:
            return cost_matrix
        
        observer_pos_global = observer_state_global[:2].flatten()

        for i, track in enumerate(tracks):
            track_pred_pos_global = track.get_predicted_position()
            dist_to_observer = np.linalg.norm(track_pred_pos_global - observer_pos_global)

            if dist_to_observer < 0.1: # 保护：航迹预测离观测者太近
                continue

            # z_pred_polar is a 3x1 column vector [r, theta_local, vr]^T
            z_pred_polar_col_vec = track.get_predicted_measurement_polar(observer_state_global)
            S = track.get_innovation_covariance(observer_state_global)

            if z_pred_polar_col_vec is None or S is None: # hx 或 H 计算失败
                continue

            try:
                S_inv = np.linalg.inv(S)
            except np.linalg.LinAlgError: # S 矩阵奇异
                # print(f"Warning: Singular S matrix for track {track.track_id}. S:\n{S}")
                continue

            for j, meas_polar in enumerate(current_polar_measurements):
                z_actual_col_vec = np.array(meas_polar).reshape(3, 1)
                
                y = z_actual_col_vec - z_pred_polar_col_vec
                y[1, 0] = (y[1, 0] + np.pi) % (2 * np.pi) - np.pi # 角度差归一化

                mahalanobis_sq_dist = y.T @ S_inv @ y
                
                # cfg.GATING_THRESHOLD 应该是卡方分布的阈值
                if mahalanobis_sq_dist < cfg.GATING_THRESHOLD:
                    cost_matrix[i, j] = mahalanobis_sq_dist
                    
        return cost_matrix

    def _data_association_gnn(self, cost_matrix):
        if cost_matrix.shape[0] == 0 or cost_matrix.shape[1] == 0 or np.all(cost_matrix == np.inf):
            return [], list(range(cost_matrix.shape[0])), list(range(cost_matrix.shape[1]))

        row_ind, col_ind = linear_sum_assignment(cost_matrix)
        matched_pairs = []
        unmatched_track_indices_set = set(range(cost_matrix.shape[0]))
        unmatched_detection_indices_set = set(range(cost_matrix.shape[1]))

        for r, c in zip(row_ind, col_ind):
            if cost_matrix[r, c] < np.inf:
                matched_pairs.append((self.tracks[r], c))
                unmatched_track_indices_set.discard(r)
                unmatched_detection_indices_set.discard(c)
        
        unmatched_tracks = [self.tracks[i] for i in unmatched_track_indices_set]
        return matched_pairs, unmatched_tracks, list(unmatched_detection_indices_set)

    def initiate_track(self, detection_polar, observer_state_global, current_time):
        new_track = Track(detection_polar, observer_state_global, current_time, self.dt)
        self.tracks.append(new_track)
        # print(f"Time {current_time:.1f}s: Initiated Track ID {new_track.track_id} from detection [r:{detection_polar[0]:.1f}, th:{np.rad2deg(detection_polar[1]):.1f}deg, vr:{detection_polar[2]:.1f}]. Total tracks: {len(self.tracks)}")

    def manage_tracks(self, current_time):
        tracks_to_keep = []
        for track in self.tracks:
            if track.state == 'Tentative':
                if track.hits >= cfg.M_CONFIRM:
                    track.state = 'Confirmed'
                    # print(f"Time {current_time:.1f}s: Track ID {track.track_id} Confirmed. Hits: {track.hits}, Age: {track.age}")
                elif track.age > cfg.N_CONFIRM and track.hits < cfg.M_CONFIRM : # 修正：应该是 track.age > N_CONFIRM
                    track.state = 'Deleted'
                    # print(f"Time {current_time:.1f}s: Tentative Track ID {track.track_id} Deleted (age {track.age} > N_CONFIRM {cfg.N_CONFIRM}, hits {track.hits} < M_CONFIRM {cfg.M_CONFIRM}).")

            if track.misses > cfg.MAX_CONSECUTIVE_MISSES:
                if track.state != 'Deleted':
                    track.state = 'Deleted'
                    # print(f"Time {current_time:.1f}s: Track ID {track.track_id} (was {track.state if track.state!='Deleted' else 'Tentative/Confirmed'}) Deleted (max misses {track.misses}).")

            if track.state != 'Deleted':
                tracks_to_keep.append(track)
        
        self.tracks = tracks_to_keep


    def step(self, detected_targets_global_cartesian, raw_measurements_polar, observer_state_global, current_time):
        for track in self.tracks:
            track.predict()

        cost_m = self._calculate_cost_matrix_mahalanobis(self.tracks, raw_measurements_polar, observer_state_global)
        
        matched_pairs, unmatched_tracks_list, unmatched_detections_indices_list = \
            self._data_association_gnn(cost_m)

        for track, det_idx in matched_pairs:
            actual_measurement_polar = raw_measurements_polar[det_idx]
            track.update(actual_measurement_polar, observer_state_global, current_time)

        for track in unmatched_tracks_list:
            track.mark_missed()

        for det_idx in unmatched_detections_indices_list:
            measurement_to_init = raw_measurements_polar[det_idx]
            r_det, theta_det, _ = measurement_to_init
            if cfg.RADAR_MIN_RANGE <= r_det <= cfg.RADAR_MAX_RANGE and abs(theta_det) <= cfg.RADAR_FOV_RAD / 2:
                self.initiate_track(measurement_to_init, observer_state_global, current_time)

        self.manage_tracks(current_time)

        current_estimated_tracks_for_plot = []
        for track in self.tracks:
            if track.state != 'Deleted':
                current_estimated_tracks_for_plot.append({
                    'id': track.track_id,
                    'state_vec': track.kf.x.flatten().tolist(),
                    'history_states': np.array(track.history).tolist(),
                    'color': cfg.TARGET_COLORS[(track.track_id -1) % len(cfg.TARGET_COLORS)],
                    'current_state_str': track.state
                })
        return current_estimated_tracks_for_plot
