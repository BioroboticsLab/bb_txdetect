import numpy as np
import matplotlib.pyplot as plt
#from mpl_toolkits.mplot3d import Axes3D
import matplotlib
import scipy
import scipy.signal
import time
import datetime
import csv
import pandas
import copy
import pickle
import json
import math

from IPython.display import display, HTML

from sklearn.externals import joblib
from sklearn.pipeline import make_pipeline, make_union
import sklearn.linear_model
import sklearn.dummy
import sklearn.metrics
import sklearn.ensemble
import sklearn.calibration
import sklearn.decomposition
#import xgboost
import sklearn.preprocessing
import sklearn.model_selection
#from tqdm import tqdm_notebook
from tqdm import tqdm
import seaborn as sns
import psycopg2
from io import StringIO
import requests
import itertools

import bb_binary
import bb_binary.repository


def setSnsStyle(style):
    sns.set(style=style, font_scale=1.5)
    font = {'family': 'serif',
            'weight': 'normal',
            'size': 30}
    matplotlib.rc('font', **font)
    matplotlib.rcParams['xtick.labelsize'] = 16
    matplotlib.rcParams['ytick.labelsize'] = 16
    matplotlib.rcParams['axes.titlesize'] = 16
    matplotlib.rcParams['axes.labelsize'] = 16


setSnsStyle("ticks")


def relative_rotation(a, z_rot_a, b):
    """
    relative_rotation calculates the relative rotation of one tag to the position of another tag.
    It takes two vecors representing positions of tags and the z_rotation of the first tag.
    It returns the scalar product between the normalized vector pointing from a to b and a normalized vector
    representing the first tag's rotation.
    Assumption: z_rot_a is given in radians with value 0 pointing east, 1/2*pi south, -1/2*pi north and 1*pi west.
    """
    rotation = np.array((np.cos(z_rot_a), np.sin(z_rot_a)))
    normalized_rotation = rotation / np.linalg.norm(rotation)
    normalized_a_to_b = (b - a) / np.linalg.norm(b - a)
    return np.dot(normalized_a_to_b, normalized_rotation)


db = psycopg2.connect(
    "dbname='beesbook' user='reader' host='localhost' password='reader'")

gt_data_url = 'https://docs.google.com/spreadsheet/ccc?key=1XfezoGQK77DvKVvHwK6QDlsfbfcEiy8GjS3BLFn7UEw&output=csv'
raw_data = requests.get(gt_data_url).content.decode('Utf-8')
gt_data = pandas.read_csv(StringIO(raw_data), index_col=0)

gt_data = gt_data[gt_data.human_decidable_interaction == "y"]

# Parse event metadata from table.
class Event(object):
    track_ids = (None, None)
    detection_ids = (None, None)
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

		# TODO shouldn't duplicates get filtered out here?
    @property
    def frame_ids(self):
        for detection_id in itertools.chain(*self.detection_ids):
            yield int(detection_id[1:].split("d")[0])

gt_events = []
for i in tqdm(range(gt_data.shape[0])):
    gt_events.append(Event(gt_data.iloc[i, :]))
print("Ground truth events loaded: {}".format(len(gt_events)))

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
    frame_fc_map = pandas.DataFrame(frame_to_fc_map,
                                    columns=("frame_id", "fc_id", "fc_path", "video_name"))

    return frame_fc_map


all_frame_ids = set()
for event in gt_events:
    for frame_id in event.frame_ids:
        all_frame_ids.add(frame_id)
print("Unique frame ids: {}".format(len(all_frame_ids)))

frame_fc_map = get_frame_container_info_for_frames(db, list(all_frame_ids))


def load_frame_container(fname):
    """Loads :obj:`.FrameContainer` from this filename."""
    with open(fname, 'rb') as f:
        return bb_binary.FrameContainer.read(f, traversal_limit_in_words=2**63)


fc_files = {}
for unique_fc in np.unique(frame_fc_map.fc_path.values):
    fc_files[unique_fc] = load_frame_container(unique_fc)

frame_to_fc_map = {}
for fc_path, df in tqdm(frame_fc_map.groupby("fc_path")):
    for frame in df.frame_id.values:
        frame_to_fc_map[frame] = fc_files[fc_path]

# For every event, map additional data.
# With the frame container / frame, we can now load all the original
# bb_binary data for the detections.
for event in tqdm(gt_events):
    beecoords = ([], [])
    ts_set = set()
    for bee in range(len(event.detection_ids)):
        for detection_id in event.detection_ids[bee]:
            frame_id, detection_idx = detection_id[1:].split("d")
            frame_id = int(frame_id)
            detection_idx = int(detection_idx.split("c")[0])
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

    abee = pandas.DataFrame(
        beecoords[0],
        columns=(
            "x1",
            "y1",
            "orient1",
            "timestamp1"))
    bbee = pandas.DataFrame(
        beecoords[1],
        columns=(
            "x2",
            "y2",
            "orient2",
            "timestamp2"))

    event.df = pandas.concat((abee, bbee), ignore_index=True, axis=1)
    event.df.columns = list(abee.columns) + list(bbee.columns)
    assert len(ts_set) == 0

event_df = pandas.concat([event.df for event in gt_events], axis=0, ignore_index=True)

for event in tqdm(gt_events):
    event.df["trophallaxis_observed"] = event.trophallaxis_observed
        
verified_events = list(filter(lambda x: x.trophallaxis_observed == True, gt_events))

verified = gt_data[gt_data.trophallaxis_observed == 'y']
durations = verified['trophallaxis_end_frame_nr'] - verified['trophallaxis_start_frame_nr'] + 1

print("done")
