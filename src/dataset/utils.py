from pathlib import Path

import numpy as np
import pandas as pd


def get_game_meta_data(game: str, config: dict) -> pd.DataFrame:
    """
    Reads meta data CSV from config and returns DataFrame grouped by subject_id with list of trial_ids for the given game.
    """
    meta_data_path = Path(config["meta_data_path"])
    meta_data = pd.read_csv(meta_data_path)
    game_meta_data = (
        meta_data[meta_data["GameName"] == game]
        .groupby("subject_id")["trial_id"]
        .apply(list)
        .reset_index()
    )
    return game_meta_data


def get_train_test_files(game, config):
    game_meta_data = get_game_meta_data(game, config)
    train_files = []
    test_files = []
    read_path = config["processed_data_path"] + f"{game}/"

    for _, row in game_meta_data.iterrows():
        subject_id = row["subject_id"]
        trial_ids = row["trial_id"]
        name = [
            read_path + subject_id + "_" + str(trial_id) + ".tar.gz"
            for trial_id in trial_ids
        ]
        if subject_id in config["train_sub"]:
            train_files.extend(name)

        if subject_id in config["test_sub"]:
            test_files.extend(name)

    return train_files, test_files


class OnlineFixationDetector:
    def __init__(self, maxdist=25, mindur=5, missing=0.0):
        """
        Parameters
        ----------
        maxdist : float
            Maximum distance from fixation centroid (pixels)
        mindur : int
            Minimum number of frames for a valid fixation
        missing : float
            Missing data value
        """
        self.maxdist = maxdist
        self.mindur = mindur
        self.missing = missing

        self.current_fixation = []
        self.fixations = []
        self.frame_idx = 0

    def update(self, gaze_xy):
        """
        Process a single frame.

        Parameters
        ----------
        gaze_xy : np.array or list
            [x, y] gaze coordinate for one frame
        """

        self.frame_idx += 1

        # Handle missing data
        if gaze_xy[0] == self.missing or gaze_xy[1] == self.missing:
            self._close_fixation()
            return

        gaze_xy = np.asarray(gaze_xy)

        if not self.current_fixation:
            self.current_fixation.append(gaze_xy)
            return

        # Compute centroid of current fixation
        fixation_array = np.array(self.current_fixation)
        centroid = fixation_array.mean(axis=0)

        distance = np.linalg.norm(gaze_xy - centroid)

        if distance <= self.maxdist:
            # Still part of the same fixation
            self.current_fixation.append(gaze_xy)
        else:
            # Fixation ended
            self._close_fixation()
            self.current_fixation.append(gaze_xy)

    def _close_fixation(self):
        if len(self.current_fixation) >= self.mindur:
            fixation_array = np.array(self.current_fixation)
            self.fixations.append(
                {
                    "end_frame": self.frame_idx - 1,
                    "duration": len(self.current_fixation),
                    "x_mean": fixation_array[:, 0].mean(),
                    "y_mean": fixation_array[:, 1].mean(),
                    "count": len(self.current_fixation),
                }
            )
        self.current_fixation = []

    def finalize(self):
        """Call once after the last frame"""
        self._close_fixation()
        return self.fixations

    import numpy as np


class OnlineClusterer:
    def __init__(self, maxdist=25):
        self.maxdist = maxdist
        self.clusters = []

    def update(self, point):
        point = np.asarray(point)

        if not self.clusters:
            self._new_cluster(point)
            return

        distances = [np.linalg.norm(point - c["centroid"]) for c in self.clusters]

        idx = np.argmin(distances)

        if distances[idx] <= self.maxdist:
            self._update_cluster(idx, point)
        else:
            self._new_cluster(point)

    def _new_cluster(self, point):
        self.clusters.append({"centroid": point.astype(float), "count": 1})

    def _update_cluster(self, idx, point):
        c = self.clusters[idx]
        n = c["count"]
        c["centroid"] += (point - c["centroid"]) / (n + 1)
        c["count"] += 1
