import numpy as np
import pandas as pd
import json
from tqdm import tqdm
import seaborn as sns
import psycopg2
import bb_binary
from psycopg2.extensions import connection

from typing import List, Dict

class Event(object):
    track_ids = (None, None)
    detection_ids = (None, None)
    bee_ids = (None, None)
    begin_frame_idx, end_frame_idx = np.NaN, np.NaN
    trophallaxis_observed = False

    def __init__(self, row):
        self.detection_ids = (
            json.loads(row.bee_a_detection_ids.replace("'", "\"")),
            json.loads(row.bee_b_detection_ids.replace("'", "\"")))
        self.track_ids = json.loads(
            row.track_id_combination
            .replace("'", "\"")
            .replace("(", "[")
            .replace(")", "]"))
        self.begin_frame_idx = row.trophallaxis_start_frame_nr
        self.end_frame_idx = row.trophallaxis_end_frame_nr
        self.trophallaxis_observed = row.trophallaxis_observed == 'y'

    @property
    def frame_ids(self):
		# both bees have always the same frame_ids
        for detection_id in self.detection_ids[0]:
            yield int(detection_id[1:].split("d")[0])


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
    return psycopg2.connect(
        "dbname='beesbook' user='reader' host='localhost' password='reader'")


def load_gt_data():
    gt_data = pd.read_csv('csv/ground_truth_concat.csv', index_col=0)
    gt_data = gt_data[gt_data.human_decidable_interaction == "y"]

    gt_events = []
    for i in tqdm(range(gt_data.shape[0])):
        gt_events.append(Event(gt_data.iloc[i, :]))
    print("Ground truth events loaded: {}".format(len(gt_events)))
    return gt_events, gt_data


# Map frame container info to event frames.
# The frame container will be used to load the positional data.
def get_frame_container_info_for_frames(database, frame_ids):
    cur = database.cursor()
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
def load_frame_container(fname):
    """Loads :obj:`.FrameContainer` from this filename."""
    with open(fname, 'rb') as f:
        return bb_binary.FrameContainer.read(f, traversal_limit_in_words=2**63)


def get_all_frame_ids(gt_events):
    all_frame_ids = set()
    for event in gt_events:
        for frame_id in event.frame_ids:
            all_frame_ids.add(frame_id)
    print("Unique frame ids: {}".format(len(all_frame_ids)))
    return all_frame_ids


def get_frame_to_fc_path_dict(frame_fc_map : pd.DataFrame) -> Dict:
    fc_files = {}
    for unique_fc in np.unique(frame_fc_map.fc_path.values):
        fc_files[unique_fc] = load_frame_container(unique_fc)

    frame_to_fc_map = {}
    for fc_path, df in tqdm(frame_fc_map.groupby("fc_path")):
        for frame in df.frame_id.values:
            frame_to_fc_map[frame] = fc_files[fc_path]
    return frame_to_fc_map


def split_detection_id(detection_id: str) -> (int, int):
    frame_id, detection_idx = detection_id[1:].split("d")
    frame_id = int(frame_id)
    detection_idx = int(detection_idx.split("c")[0])
    return frame_id, detection_idx


def map_additional_data_to_events(gt_events: List[Event], frame_to_fc_map):
    """For every event, map additional data.
    With the frame container / frame, we can now load all the original
    bb_binary data for the detections."""
    for event in tqdm(gt_events):
        beecoords = ([], [])
        ts_set = set()
        for bee in range(len(event.detection_ids)):
            for detection_id in event.detection_ids[bee]:
                frame_id, detection_idx = split_detection_id(detection_id)
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


def map_bee_ids(db: connection, events: List[Event]):
    for event in events:
        for i in range(2):
            event.bee_ids[i] = get_bee_id(db, *split_detection_id(event.detection_ids[i]))
    

def get_bee_id(db: connection, frame_id: int, detection_idx: int):
    cur = db.cursor()
    query = "select bee_id from bb_detections where frame_id = {} and detection_idx = {};".format(frame_id, detection_idx);
    cur.execute(query)
    result = cur.fetchone()
    if result:
        return result[0]
    else:
        return -1


def get_timestamp(db: connection, frame_id: int) -> str:
    cur = db.cursor()
    cur.execute("SELECT * FROM bb_detections where frame_id = {}".format(frame_id))
    return str(cur.fetchone()[0])


def get_frames_before_after(db: connection, frame_id: int, bee_ids: (int, int), num_frames: int, before: bool = True):
    timestamp = get_timestamp(db=db, frame_id=frame_id)
    
    if before:
        ltgt = "<" 
        order = "desc"
    else:
        ltgt = ">" 
        order = "asc"
        
    cur = db.cursor()
        
    query =  """SELECT timestamp, frame_id, x_pos, y_pos, orientation, bee_id FROM bb_detections 
    where timestamp {} '{}' 
    and (bee_id = {} or bee_id = {})
    order by timestamp desc 
    limit {}""".format(ltgt, timestamp, *bee_ids, num_frames * 2)

    cur.execute(query)

    rows = []
    for timestamp, frame_id, x_pos, y_pos, orientation, bee_id in cur: 
        rows.append((timestamp, frame_id, x_pos, y_pos, orientation, bee_id))
    return pd.DataFrame(rows, columns=("timestamp", "frame_id", "x_pos", "y_pos", "orientation", "bee_id"))


