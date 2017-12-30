import numpy as np
import pandas as pd
import matplotlib
import json
from tqdm import tqdm
import seaborn as sns
import psycopg2
import bb_binary
import numba
import math

from psycopg2.extensions import connection
from typing import List, Dict


class Event(object):
    track_ids = (None, None)
    detection_ids = (None, None)
    bee_ids = (None, None)
    begin_frame_idx, end_frame_idx = -1, -1
    trophallaxis_observed = False
    df = None
    frames_ids_before = None
    frames_before = None
    frames_ids_after = None
    frames_after = None
    not_nans_before = None
    not_nans_after = None

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


def np_float_to_int(x: np.float) -> int:
    if(np.isnan(x)):
        return -1
    else:
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

    def connect(self):
        self.db = psycopg2.connect("dbname='beesbook' user='reader' host='localhost' password='reader'")


    def load_gt_data(self, path: str = 'csv/ground_truth_concat.csv'):
        gt_data = pd.read_csv(path, index_col=0)
        gt_data = gt_data[gt_data.human_decidable_interaction == "y"]

        self.gt_events = []
        for i in tqdm(range(gt_data.shape[0])):
            self.gt_events.append(Event(gt_data.iloc[i, :]))
        tqdm.write("Ground truth events loaded: {}".format(len(self.gt_events)))


    # Map frame container info to event frames.
    # The frame container will be used to load the positional data.
    def get_frame_container_info_for_frames(self, frame_ids):
        cur = self.db.cursor()

        cur.execute(
            "SELECT fc_id, frame_id FROM plotter_frame WHERE frame_id IN %s;",
            (tuple(frame_ids),))

        frame_container_to_frames = {}
        for fc_id, frame_id in tqdm(cur):
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
        for fc_path, df in tqdm(frame_fc_map.groupby("fc_path")):
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
        for event in tqdm(self.gt_events):
            beecoords = ([], [])
            ts_set = set()
            for bee in range(len(event.detection_ids)):
                for detection_id in event.detection_ids[bee]:
                    frame_id, detection_idx = self.split_detection_id(detection_id)
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
        for event in tqdm(self.gt_events):
            event.bee_ids = (self.get_bee_id(*self.split_detection_id(event.detection_ids[0][0])),
                             self.get_bee_id(*self.split_detection_id(event.detection_ids[1][0])))



    def get_bee_id(self, frame_id: int, detection_idx: int):
        cur = self.db.cursor()
        if not self.get_bee_id_is_prepared:
            cur.execute("PREPARE get_bee_id AS SELECT bee_id FROM bb_detections_subset WHERE frame_id = $1 AND detection_idx = $2;")
            self.get_bee_id_is_prepared = True
        cur.execute("EXECUTE get_bee_id (%s, %s);", (frame_id, detection_idx))
        result = cur.fetchone()
        if result:
            return result[0]
        else:
            return -1


    def get_timestamp(self, frame_id: int):
        cur = self.db.cursor()
        cur.execute("SELECT timestamp FROM plotter_frame where frame_id = %s", (frame_id,))
        return cur.fetchone()[0]


    def map_frames_before_after(self, num_frames: int):
        tqdm.write("map frames before and after")
        for event in tqdm(self.gt_events):
            frame_ids = list(event.frame_ids)
            self.set_frames_before_after(frame_id_begin=frame_ids[0], 
                                         frame_id_end=frame_ids[len(list(event.frame_ids))-1],
                                         bee_ids=event.bee_ids, 
                                         num_frames=num_frames, 
                                         event=event)


    def get_position_and_orientation(self, frame_id: int, bee_ids: (int, int)):
        cursor = self.db.cursor()
        cursor.execute("SELECT bee_id, x_pos, y_pos, orientation FROM bb_detections_subset where frame_id = %s and (bee_id = %s or bee_id = %s)",
                       (frame_id, *bee_ids))
        return list(cursor)



    def set_frames_before_after(self, frame_id_begin: int, frame_id_end: int, bee_ids: (int, int), num_frames: int, event: Event):

        timestamp_begin = self.get_timestamp(frame_id=frame_id_begin)
        timestamp_end = self.get_timestamp(frame_id=frame_id_end)
        seconds = num_frames / 3

        cursor = self.db.cursor()
        cursor.execute("SELECT frame_id FROM plotter_frame WHERE timestamp >= %s AND timestamp < %s", 
                       (timestamp_begin - seconds, timestamp_begin))

        event.frame_ids_before, event.frames_before, event.not_nans_a_before, event.not_nans_b_before= self.interpolate(cursor, bee_ids)
        
        cursor.execute("SELECT frame_id FROM plotter_frame WHERE timestamp > %s AND timestamp <= %s", 
                       (timestamp_end, timestamp_end + seconds))

        event.frame_ids_after, event.frames_after, event.not_nans_a_after, event.not_nans_b_after= self.interpolate(cursor, bee_ids)




    def interpolate(self, cursor, bee_ids):
        results = np.empty((cursor.rowcount, 6), dtype=np.float32)
        results[:,:] = np.nan
        frame_ids = []
        i = 0
        for row in cursor:
            frame_id = row[0]
            detections = self.get_position_and_orientation(frame_id=frame_id, bee_ids=bee_ids)
            frame_ids.append(frame_id)
            for d in detections:
                if d[0] == bee_ids[0]:
                    results[i,:3] = [d[1], d[2], d[3]]
                else:
                    results[i,3:] = [d[1], d[2], d[3]]
            i += 1

        # now results may have gaps with nans, but all frames are included

        not_nans_a = interpolate_trajectory(results[:,:3])
        not_nans_b = interpolate_trajectory(results[:,3:])

        return (frame_ids, results, not_nans_a, not_nans_b)
        

@numba.njit
def short_angle_dist(a0,a1):
    max = math.pi*2
    da = (a1 - a0) % max
    return 2*da % max - da


@numba.njit(numba.float32[:](numba.float32[:, :]))
def interpolate_trajectory(trajectory):
   
    nans = np.isnan(trajectory[:,0])
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
            begin_t -= 1 # extrapolate right
        elif begin_t == -1:
            begin_t = 0 # extrapolate left
       
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
