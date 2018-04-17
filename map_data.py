import math
import json
from typing import List, Dict
import numpy as np
import pandas as pd
import matplotlib
from tqdm import tqdm
import seaborn as sns
import psycopg2
import numba
import bb_binary


class Event(object):
    track_ids = (None, None)
    detection_ids = (None, None)
    bee_ids = (None, None)
    begin_frame_idx, end_frame_idx = -1, -1
    trophallaxis_observed = False
    df = None
    observations = []

    def __init__(self, row):
        self.detection_ids = (
            json.loads(row.bee_a_detection_ids.replace("'", "\"")),
            json.loads(row.bee_b_detection_ids.replace("'", "\"")))
        self.track_ids = json.loads(
            row.track_id_combination
            .replace("'", "\"")
            .replace("(", "[")
            .replace(")", "]"))
        self.begin_frame_idx = np_float_to_int(row.trophallaxis_start_frame_nr)
        self.end_frame_idx = np_float_to_int(row.trophallaxis_end_frame_nr)
        self.trophallaxis_observed = row.trophallaxis_observed == 'y'

    @property
    def frame_ids(self):
        # both bees have always the same frame_ids
        for detection_id in self.detection_ids[0]:
            yield int(detection_id[1:].split("d")[0])


class Observation(object):
    def __init__(self, frame_id, xs, ys, orientations):
        self.frame_id = frame_id
        self.xs = [int(x) for x in xs]
        self.ys = [int(y) for y in ys]
        self.orientations = orientations
        self.trophallaxis_observed = None

    @property
    def file_name(self) -> str:
        if self.trophallaxis_observed is None:
            label = "u"
        elif self.trophallaxis_observed:
            label = "y"
        else:
            label = "n"
        return "{}_{}_{}_{}_{}_{}_{}_{}".format(self.frame_id,
                                                *self.xs,
                                                *self.ys,
                                                *[int((math.degrees(o) + 360) % 360) for o in self.orientations],
                                                label)


def np_float_to_int(x: np.float) -> int:
    if(np.isnan(x)):
        return -1
    return int(x)


def setSnsStyle(style: str):
    # set to 'ticks', to not have lines in the images
    sns.set(style=style, font_scale=1.5)
    font = {'family': 'serif',
            'weight': 'normal',
            'size': 30}
    matplotlib.rc('font', **font)
    matplotlib.rcParams['xtick.labelsize'] = 16
    matplotlib.rcParams['ytick.labelsize'] = 16
    matplotlib.rcParams['axes.titlesize'] = 16
    matplotlib.rcParams['axes.labelsize'] = 16


class DataMapper:
    get_bee_id_is_prepared = False
    db = None
    hide_progress_bars = True

    def __init__(self, path=None, hide_progress_bars=False):
        setSnsStyle("ticks")
        self.hide_progress_bars = hide_progress_bars
        self.connect()
        if not path:
            return
        self.load_gt_data(path=path)
        frame_fc_map = self.get_frame_container_info_for_frames(list(self.get_all_frame_ids()))
        frame_to_fc_map = self.get_frame_to_fc_path_dict(frame_fc_map)
        self.map_additional_data_to_events(frame_to_fc_map)
        self.map_bee_ids()
        self.map_observations(padding=8)

    def connect(self):
        self.db = psycopg2.connect(
            "dbname='beesbook' user='reader' host='localhost' password='reader'")

    def load_gt_data(self, path: str):
        gt_data = pd.read_csv(path, index_col=0)
        gt_data = gt_data[gt_data.human_decidable_interaction == "y"]

        self.gt_events = []
        for i in tqdm(range(gt_data.shape[0]), disable=self.hide_progress_bars):
            self.gt_events.append(Event(gt_data.iloc[i, :]))
        tqdm.write("Ground truth events loaded: {}".format(
            len(self.gt_events)))

    # Map frame container info to event frames.
    # The frame container will be used to load the positional data.
    def get_frame_container_info_for_frames(self, frame_ids):
        cur = self.db.cursor()

        cur.execute(
            "SELECT fc_id, frame_id FROM plotter_frame WHERE frame_id IN %s;",
            (tuple(frame_ids),))

        frame_container_to_frames = {}
        for fc_id, frame_id in tqdm(cur, disable=self.hide_progress_bars):
            if fc_id not in frame_container_to_frames:
                frame_container_to_frames[fc_id] = []
            frame_container_to_frames[fc_id].append(frame_id)

        cur.execute(
            "SELECT id, fc_id, fc_path, video_name FROM plotter_framecontainer WHERE id IN %s;",
            (tuple(frame_container_to_frames.keys()),))

        frame_to_fc_map = []
        for ID, fc_id, fc_path, video_name in cur:
            for frame_id in frame_container_to_frames[ID]:
                frame_to_fc_map.append((frame_id, fc_id, fc_path, video_name))
        frame_fc_map = pd.DataFrame(frame_to_fc_map,
                                    columns=("frame_id", "fc_id", "fc_path", "video_name"))
        return frame_fc_map

    # TODO replace bb_binary
    def load_frame_container(self, fname):
        """Loads :obj:`.FrameContainer` from this filename."""
        with open(fname, 'rb') as f:
            return bb_binary.FrameContainer.read(f, traversal_limit_in_words=2**63)

    def get_all_frame_ids(self):
        all_frame_ids = set()
        for event in self.gt_events:
            for frame_id in event.frame_ids:
                all_frame_ids.add(frame_id)
        tqdm.write("Unique frame ids: {}".format(len(all_frame_ids)))
        return all_frame_ids

    def get_frame_to_fc_path_dict(self, frame_fc_map: pd.DataFrame) -> Dict:
        fc_files = {}
        for unique_fc in np.unique(frame_fc_map.fc_path.values):
            fc_files[unique_fc] = self.load_frame_container(unique_fc)

        frame_to_fc_map = {}
        for fc_path, df in tqdm(frame_fc_map.groupby("fc_path"), disable=self.hide_progress_bars):
            for frame in df.frame_id.values:
                frame_to_fc_map[frame] = fc_files[fc_path]
        return frame_to_fc_map

    def split_detection_id(self, detection_id: str) -> (int, int):
        frame_id, detection_idx = detection_id[1:].split("d")
        frame_id = int(frame_id)
        detection_idx = int(detection_idx.split("c")[0])
        return frame_id, detection_idx

    def map_additional_data_to_events(self, frame_to_fc_map):
        """For every event, map additional data.
        With the frame container / frame, we can now load all the original
        bb_binary data for the detections."""
        for event in tqdm(self.gt_events, disable=self.hide_progress_bars):
            beecoords = ([], [])
            ts_set = set()
            for bee in range(len(event.detection_ids)):
                for detection_id in event.detection_ids[bee]:
                    frame_id, detection_idx = self.split_detection_id(
                        detection_id)
                    fc = frame_to_fc_map[frame_id]
                    frame = None

                    for frame in fc.frames:
                        if frame.id != frame_id:
                            continue
                        break
                    assert frame is not None
                    assert frame.id == frame_id

                    detection = frame.detectionsUnion.detectionsDP[detection_idx]
                    beecoords[bee].append(
                        (detection.xpos,
                         detection.ypos,
                         detection.zRotation,
                         frame.timestamp))

                    # Plausibility.
                    if frame.timestamp in ts_set:
                        ts_set.remove(frame.timestamp)
                    else:
                        ts_set.add(frame.timestamp)

            abee = pd.DataFrame(
                beecoords[0],
                columns=(
                    "x1",
                    "y1",
                    "orient1",
                    "timestamp1"))
            bbee = pd.DataFrame(
                beecoords[1],
                columns=(
                    "x2",
                    "y2",
                    "orient2",
                    "timestamp2"))

            event.df = pd.concat((abee, bbee), ignore_index=True, axis=1)
            event.df.columns = list(abee.columns) + list(bbee.columns)
            assert len(ts_set) == 0

    def map_bee_ids(self):
        tqdm.write("map bee ids")
        for event in tqdm(self.gt_events, disable=self.hide_progress_bars):
            event.bee_ids = (self.get_bee_id(*self.split_detection_id(event.detection_ids[0][0])),
                             self.get_bee_id(*self.split_detection_id(event.detection_ids[1][0])))

    def get_bee_id(self, frame_id: int, detection_idx: int):
        cur = self.db.cursor()
        if not self.get_bee_id_is_prepared:
            cur.execute(
                "PREPARE get_bee_id AS SELECT bee_id FROM bb_detections WHERE frame_id = $1 AND detection_idx = $2;")
            self.get_bee_id_is_prepared = True
        cur.execute("EXECUTE get_bee_id (%s, %s);", (frame_id, detection_idx))
        result = cur.fetchone()
        if result:
            return result[0]
        return -1

    def map_observations(self, padding: int):
        tqdm.write("map frames before and after")
        for event in tqdm(self.gt_events, disable=self.hide_progress_bars):
            frame_ids = list(event.frame_ids)
            event.observations = self.get_all_frames(frame_id_begin=frame_ids[0],
                                                     frame_id_end=frame_ids[len(
                                                         list(event.frame_ids))-1],
                                                     bee_ids=event.bee_ids,
                                                     frame_padding_length=padding)
            event.padding = padding
            for i, obs in enumerate(event.observations):
                if i < padding or i >= len(event.observations) - padding:
                    obs.trophallaxis_observed = None
                elif i >= event.begin_frame_idx and i <= event.end_frame_idx:
                    obs.trophallaxis_observed = True
                else:
                    obs.trophallaxis_observed = False

    def get_position_and_orientation(self, frame_id: int, bee_ids: (int, int)):
        cursor = self.db.cursor()
        cursor.execute("SELECT bee_id, x_pos, y_pos, orientation FROM bb_detections WHERE frame_id = %s and (bee_id = %s or bee_id = %s)",
                       (frame_id, *bee_ids))
        return list(cursor)

    def get_timestamp(self, frame_id: int):
        cur = self.db.cursor()
        cur.execute(
            "SELECT timestamp FROM plotter_frame where frame_id = %s LIMIT 1", (frame_id,))
        return cur.fetchone()[0]

    def get_neighbour_frames(self, frame_id1, frame_id2, n_frames=None, seconds=None, mode: str = 'around'):

        with psycopg2.connect("dbname='beesbook' user='reader' host='localhost' password='reader'",
                              application_name="get_neighbour_frames") as db:

            seconds = seconds or (n_frames / 3 if n_frames else 5.0)
            timestamp1 = self.get_timestamp(frame_id1)
            timestamp2 = self.get_timestamp(frame_id2)

            if mode == 'before':
                ts1 = timestamp1 - seconds
                ts2 = timestamp2
            if mode == 'after':
                ts1 = timestamp1
                ts2 = timestamp2 + seconds
            if mode == 'around':
                ts1 = timestamp1 - seconds
                ts2 = timestamp2 + seconds

            cursor = db.cursor()
            cursor.execute(
                "SELECT fc_id FROM plotter_frame WHERE frame_id = %s LIMIT 1", (frame_id1,))
            fc_id = cursor.fetchone()[0]

            cursor.execute(
                "SELECT timestamp, frame_id, fc_id FROM plotter_frame WHERE timestamp >= %s AND timestamp <= %s", (ts1, ts2))
            results = list(cursor)
            containers = {fc_id for (_, _, fc_id) in results}

            cursor.execute("PREPARE fetch_container AS "
                           "SELECT CAST(SUBSTR(video_name, 5, 1) AS INT) FROM plotter_framecontainer "
                           "WHERE id = $1")
            cursor.execute("EXECUTE fetch_container (%s)", (fc_id,))
            target_cam = cursor.fetchone()[0]
            matching_cam = set()
            for container in containers:
                cursor.execute("EXECUTE fetch_container (%s)", (container,))
                cam = cursor.fetchone()[0]
                if cam == target_cam:
                    matching_cam.add(container)
            results = [(timestamp, frame_id, target_cam) for (
                timestamp, frame_id, fc_id) in results if fc_id in matching_cam]
            return sorted(results)

    def get_all_frames(self, frame_id_begin: int, frame_id_end: int, bee_ids: (int, int), frame_padding_length: int = None):
        frames = self.get_neighbour_frames(
            frame_id1=frame_id_begin, frame_id2=frame_id_end, n_frames=frame_padding_length, mode='around')
        frame_ids = [frame_id for (timestamp, frame_id, fc_id) in frames]
        interpolated = self.interpolate(frame_ids, bee_ids)
        return [Observation(frame_id=frame_ids[i],
                            xs=(interpolated[i][0], interpolated[i][3]),
                            ys=(interpolated[i][1], interpolated[i][4]),
                            orientations=(interpolated[i][2], interpolated[i][5])) for i in range(len(frame_ids))]

    def interpolate(self, frame_ids, bee_ids):
        results = np.empty((len(frame_ids), 6), dtype=np.float32)
        results[:, :] = np.nan
        for i, frame_id in enumerate(frame_ids):
            detections = self.get_position_and_orientation(
                frame_id=frame_id, bee_ids=bee_ids)
            for d in detections:
                if d[0] == bee_ids[0]:
                    results[i, :3] = [d[1], d[2], d[3]]
                else:
                    results[i, 3:] = [d[1], d[2], d[3]]

        # now results may have gaps with nans, but all frames are included

        interpolate_trajectory(results[:, :3])
        interpolate_trajectory(results[:, 3:])
        return results


@numba.njit
def short_angle_dist(a0, a1):
    max = math.pi*2
    da = (a1 - a0) % max
    return 2*da % max - da


@numba.njit(numba.float32[:](numba.float32[:, :]))
def interpolate_trajectory(trajectory):

    nans = np.isnan(trajectory[:, 0])
    not_nans = ~nans

    nans_idx = np.where(nans)[0]
    valid_idx = np.where(not_nans)[0]
    if len(valid_idx) < 2:
        return np.zeros(shape=(trajectory.shape[0]), dtype=np.float32)

    # Interpolate gaps.
    for i in nans_idx:
        # Find closest two points to use for interpolation.
        begin_t = np.searchsorted(valid_idx, i) - 1
        if begin_t == len(valid_idx) - 1:
            begin_t -= 1  # extrapolate right
        elif begin_t == -1:
            begin_t = 0  # extrapolate left

        begin_t_idx = valid_idx[begin_t]
        end_t_idx = valid_idx[begin_t + 1]

        last_t = trajectory[begin_t_idx]
        next_t = trajectory[end_t_idx]
        dx = (end_t_idx - begin_t_idx) / 3.0
        m = [(next_t[0] - last_t[0]) / dx,
             (next_t[1] - last_t[1]) / dx,
             short_angle_dist(last_t[2], next_t[2]) / dx]

        dt = (i - begin_t_idx) / 3.0
        e = [m[i] * dt + last_t[i] for i in range(3)]
        trajectory[i] = e

    return not_nans.astype(np.float32)
