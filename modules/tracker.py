# modules/tracker.py
import numpy as np
from scipy.optimize import linear_sum_assignment
from .track import Track
import config as cfg

class Tracker:
    def __init__(self):
        self.tracks = []
        self.dt = cfg.DT

    def _calculate_cost_matrix(self, tracks, detected_clusters):
        num_tracks = len(tracks)
        num_detections = len(detected_clusters)
        cost_matrix = np.full((num_tracks, num_detections), np.inf)
        for i, track in enumerate(tracks):
            track_pos = track.get_predicted_position()
            for j, cluster in enumerate(detected_clusters):
                det_pos = cluster['position_global']
                dist = np.linalg.norm(track_pos - det_pos)
                if dist < cfg.GATING_THRESHOLD_EUCLIDEAN:
                    cost_matrix[i, j] = dist
        return cost_matrix

    def _manage_tracks(self):
        tracks_to_keep = []
        for track in self.tracks:
            if track.misses > cfg.MAX_CONSECUTIVE_MISSES: continue
            if track.state == 'Tentative' and track.hits >= cfg.M_CONFIRM:
                track.state = 'Confirmed'
            if track.state == 'Tentative' and track.age > cfg.N_CONFIRM_AGE: continue
            tracks_to_keep.append(track)
        self.tracks = tracks_to_keep

    def step(self, detected_clusters_info, observer_state):
        for track in self.tracks:
            track.predict()
        cost_matrix = self._calculate_cost_matrix(self.tracks, detected_clusters_info)
        track_indices, detection_indices = linear_sum_assignment(cost_matrix)
        matched_track_indices = set()
        matched_detection_indices = set()
        for i, j in zip(track_indices, detection_indices):
            if cost_matrix[i, j] < cfg.GATING_THRESHOLD_EUCLIDEAN:
                cluster = detected_clusters_info[j]
                measurement_to_update = cluster['measurements_polar'][0]
                self.tracks[i].update(measurement_to_update, observer_state)
                matched_track_indices.add(i)
                matched_detection_indices.add(j)
        for i, track in enumerate(self.tracks):
            if i not in matched_track_indices:
                track.mark_missed()
        for j, cluster in enumerate(detected_clusters_info):
            if j not in matched_detection_indices:
                measurement_to_init = cluster['measurements_polar'][0]
                new_track = Track(measurement_to_init, observer_state, self.dt)
                self.tracks.append(new_track)
        self._manage_tracks()
        return [{'id': t.track_id, 'history': t.history, 'state': t.state} for t in self.tracks]