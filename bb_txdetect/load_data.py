import math
import json
from typing import Dict
import warnings
import numpy as np
import pandas as pd
import matplotlib
import seaborn as sns
import psycopg2
import numba
from bb_binary import FrameContainer


class Event():
    """
    Attributes:
        observations: of frames including padding
        trophallaxis_observed: True if at least one frame shows trophallaxis
        begin_frame_idx, end_frame_idx: begin and end of trophallaxis,
            WITHOUT padding.
    """
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


class Observation():
    """Includes all meta data about a frame to request an image file."""
    def __init__(self, frame_id, xs, ys, orientations):
        self.frame_id = frame_id
        self.xs = [int(x) for x in xs]
        self.ys = [int(y) for y in ys]
        self.orientations = orientations
        self.trophallaxis_observed = None

    @property
    def label(self) -> str:
        if self.trophallaxis_observed is None:
            return "u"
        if self.trophallaxis_observed:
            return "y"
        return "n"

    @property
    def file_name(self) -> str:
        return "{}_{}_{}_{}_{}_{}_{}_{}".format(self.frame_id,
                                                *self.xs,
                                                *self.ys,
                                                *[int((math.degrees(o) + 360)
                                                      % 360)
                                                  for o in self.orientations],
                                                self.label)


def np_float_to_int(x: np.float) -> int:
    if np.isnan(x):
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


def connect():
    return psycopg2.connect("dbname='beesbook' user='reader' host='localhost' "
                            "password='reader'")


def load_ground_truth_events(csv_path: str, padding: int) -> [Event]:
    """
    Return a list of Event objects, one Event per row in the csv file.
    Args:
        csv_path: path to the ground truth data. required columns are:
            track_id_combination, bee_a_detection_ids,bee_b_detection_ids,
            human_decidable_interaction, trophallaxis_observed,
            trophallaxis_start_frame_nr, trophallaxis_end_frame_nr
        padding: set to a positive value to add extra frames at the beginning
            and end of each event.
    """
    setSnsStyle("ticks")

    gt_data = pd.read_csv(csv_path, index_col=0)
    gt_data = gt_data[gt_data.human_decidable_interaction == "y"]

    gt_events = []
    for i in range(gt_data.shape[0]):
        gt_events.append(Event(gt_data.iloc[i, :]))
    print("Ground truth events loaded: {}".format(len(gt_events)))

    all_frame_ids = list(get_all_frame_ids(events=gt_events))
    frame_fc_map = get_frame_container_info_for_frames(frame_ids=all_frame_ids)
    print("Unique frame ids: {}".format(len(all_frame_ids)))

    map_additional_data_to_events(frame_to_fc_map=get_frame_to_fc_path_dict(frame_fc_map),
                                  events=gt_events)
    map_bee_ids(gt_events)
    map_observations(padding=padding, events=gt_events)
    return gt_events


def get_all_frame_ids(events: [Event]) -> set:
    all_frame_ids = set()
    for event in events:
        for frame_id in event.frame_ids:
            all_frame_ids.add(frame_id)
    return all_frame_ids


def map_additional_data_to_events(frame_to_fc_map: Dict, events: [Event]):
    """For every event, map additional data.
    With the frame container / frame, we can now load all the original
    bb_binary data for the detections."""
    for event in events:
        beecoords = ([], [])
        ts_set = set()
        for bee in range(len(event.detection_ids)):
            for detection_id in event.detection_ids[bee]:
                frame_id, detection_idx = split_detection_id(
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


def map_observations(padding: int, events: [Event]):
    for event in events:
        frame_ids = list(event.frame_ids)
        event.observations = get_all_frames(frame_id_begin=frame_ids[0],
                                            frame_id_end=frame_ids[len(
                                                list(event.frame_ids))-1],
                                            bee_ids=event.bee_ids,
                                            frame_padding_length=padding)
        event.padding = padding
        for i, obs in enumerate(event.observations):
            if i < padding or i >= len(event.observations) - padding:
                obs.trophallaxis_observed = None
            elif event.begin_frame_idx + padding <= i <= event.end_frame_idx + padding:
                obs.trophallaxis_observed = True
            else:
                obs.trophallaxis_observed = False


def map_bee_ids(events: [Event]):
    prepare_get_bee_id()
    for event in events:
        event.bee_ids = (get_bee_id(*split_detection_id(event.detection_ids[0][0]), prepared=True),
                         get_bee_id(*split_detection_id(event.detection_ids[1][0]), prepared=True))


def prepare_get_bee_id():
    with connect() as db:
        db.cursor().execute(
            "PREPARE get_bee_id AS SELECT bee_id FROM bb_detections "
            "WHERE frame_id = $1 AND detection_idx = $2;")


def get_bee_id(frame_id: int, detection_idx: int, prepared: bool):
    with connect() as db:
        cur = db.cursor()
        if not prepared:
            prepare_get_bee_id()
        cur.execute("EXECUTE get_bee_id (%s, %s);", (frame_id, detection_idx))
        result = cur.fetchone()
        if result:
            return result[0]
        return -1


def get_frame_container_info_for_frames(frame_ids) -> pd.DataFrame:
    """
    Map frame container info to event frames.
    The frame container will be used to load the positional data.
    Return the map as pandas DataFrame.
    """
    with connect() as db:
        cur = db.cursor()

        cur.execute(
            "SELECT fc_id, frame_id FROM plotter_frame WHERE frame_id IN %s;",
            (tuple(frame_ids),))

        frame_container_to_frames = {}
        for fc_id, frame_id in cur:
            if fc_id not in frame_container_to_frames:
                frame_container_to_frames[fc_id] = []
            frame_container_to_frames[fc_id].append(frame_id)

        cur.execute(
            "SELECT id, fc_id, fc_path, video_name "
            "FROM plotter_framecontainer WHERE id IN %s;",
            (tuple(frame_container_to_frames.keys()),))

        frame_to_fc_map = []
        for ID, fc_id, fc_path, video_name in cur:
            for frame_id in frame_container_to_frames[ID]:
                frame_to_fc_map.append((frame_id, fc_id, fc_path, video_name))
        frame_fc_map = pd.DataFrame(frame_to_fc_map,
                                    columns=("frame_id", "fc_id", "fc_path",
                                             "video_name"))
        return frame_fc_map


def get_frame_to_fc_path_dict(frame_fc_map: pd.DataFrame) -> Dict:
    fc_files = {}
    for unique_fc in np.unique(frame_fc_map.fc_path.values):
        with open(unique_fc, 'rb') as f:
            # TODO replace bb_binary
            fc_files[unique_fc] = FrameContainer.read(f, traversal_limit_in_words=2**63)

    frame_to_fc_map = {}
    for fc_path, df in frame_fc_map.groupby("fc_path"):
        for frame in df.frame_id.values:
            frame_to_fc_map[frame] = fc_files[fc_path]
    return frame_to_fc_map


def split_detection_id(detection_id: str) -> (int, int):
    frame_id, detection_idx = detection_id[1:].split("d")
    frame_id = int(frame_id)
    detection_idx = int(detection_idx.split("c")[0])
    return frame_id, detection_idx


def get_all_frames(frame_id_begin: int, frame_id_end: int, bee_ids: (int, int),
                   frame_padding_length: int = None) -> [Observation]:
    frames = get_neighbour_frames(frame_id1=frame_id_begin,
                                  frame_id2=frame_id_end,
                                  n_frames=frame_padding_length,
                                  mode='around')
    frame_ids = [frame_id for (timestamp, frame_id, fc_id) in frames]
    interpolated = interpolate(frame_ids, bee_ids)

    if np.isnan(interpolated.sum()):
        warnings.warn("interpolation failed, skipped event.")
        return []

    return [Observation(frame_id=frame_ids[i],
                        xs=(interpolated[i][0], interpolated[i][3]),
                        ys=(interpolated[i][1], interpolated[i][4]),
                        orientations=(interpolated[i][2], interpolated[i][5]))
            for i in range(len(frame_ids))]

def interpolate(frame_ids, bee_ids) -> np.ndarray:
    results = np.empty((len(frame_ids), 6), dtype=np.float32)
    results[:, :] = np.nan
    for i, frame_id in enumerate(frame_ids):
        detections = get_position_and_orientation(frame_id=frame_id,
                                                  bee_ids=bee_ids)
        for d in detections:
            if d[0] == bee_ids[0]:
                results[i, :3] = [d[1], d[2], d[3]]
            else:
                results[i, 3:] = [d[1], d[2], d[3]]

    # now results may have gaps with nans, but all frames are included

    interpolate_trajectory(results[:, :3])
    interpolate_trajectory(results[:, 3:])
    return results


def get_position_and_orientation(frame_id: int, bee_ids: (int, int)):
    with connect() as db:
        cursor = db.cursor()
        cursor.execute("SELECT bee_id, x_pos, y_pos, orientation "
                       "FROM bb_detections WHERE frame_id = %s "
                       "and (bee_id = %s or bee_id = %s)",
                       (frame_id, *bee_ids))
        return list(cursor)


def get_timestamp(frame_id: int) -> int:
    with connect() as db:
        cur = db.cursor()
        cur.execute(
            "SELECT timestamp FROM plotter_frame where frame_id = %s LIMIT 1",
            (frame_id,))
        return cur.fetchone()[0]


def get_neighbour_frames(frame_id1, frame_id2, n_frames=None, seconds=None,
                         mode: str = 'around') -> [(int, int, int)]:
    with connect() as db:
        seconds = seconds or (n_frames / 3 if n_frames else 5.0)
        timestamp1 = get_timestamp(frame_id1)
        timestamp2 = get_timestamp(frame_id2)

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
            "SELECT fc_id FROM plotter_frame WHERE frame_id = %s LIMIT 1",
            (frame_id1,))
        fc_id = cursor.fetchone()[0]

        cursor.execute(
            "SELECT timestamp, frame_id, fc_id FROM plotter_frame "
            "WHERE timestamp >= %s AND timestamp <= %s", (ts1, ts2))
        results = list(cursor)
        containers = {fc_id for (_, _, fc_id) in results}

        cursor.execute("PREPARE fetch_container AS "
                       "SELECT CAST(SUBSTR(video_name, 5, 1) AS INT) "
                       "FROM plotter_framecontainer "
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
