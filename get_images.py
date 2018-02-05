import matplotlib.pyplot as plt
import bb_backend.api
from bb_backend.api import FramePlotter, VideoPlotter
from typing import List
import matplotlib.pyplot as plt
from scipy.ndimage.interpolation import rotate
import numpy as np
import math
from tqdm import tqdm
from map_data import Event
from typing import List, Dict

FINAL_SIZE = 128
SIZE_BEFORE_ROTATION = math.ceil(FINAL_SIZE * math.sqrt(2))

debug_vals = []
debug_vals_rotation = []

def get_crop_coordinates(x1:int, y1:int, x2:int, y2: int, size: int = SIZE_BEFORE_ROTATION):
    xc = min(x1, x2) + abs(x1 - x2) // 2
    yc = min(y1, y2) + abs(y1 - y2) // 2
    global debug_vals
    debug_vals.append((xc,yc,x1,y1,x2,y2))
    return int(xc - size // 2), int(yc - size // 2), int(xc + size // 2 - 1), int(yc + size // 2 - 1)


def get_frame_plotter(frame_id: int, x1:int, y1:int, x2:int, y2: int) -> FramePlotter:
    return FramePlotter(frame_id=int(frame_id), scale=1.0, crop_coordinates=get_crop_coordinates(x1,y1,x2,y2))


def get_frame_plotter_by_event(event: Event, frame_index: int) -> FramePlotter:
    return get_frame_plotter(frame_id=list(event.frame_ids)[frame_index],
                             x1=int(event.df.x1.values[frame_index]),
                             y1=int(event.df.y1.values[frame_index]),
                             x2=int(event.df.x2.values[frame_index]),
                             y2=int(event.df.y2.values[frame_index]))


# debug
def get_center_frame_index(event: Event) -> int:
    return int(event.begin_frame_idx + (event.end_frame_idx - event.begin_frame_idx) / 2)


# debug
def get_videos_around_center(events: List[Event]):
    i = 0
    for event in tqdm(events):
        # print("start fp")
        center_frame_index = get_center_frame_index(event)
        center_fp = get_frame_plotter_by_event(event=event, frame_index=center_frame_index)
        # print("start vp")
        n = 10
        vp = VideoPlotter(frames=[center_fp],
                          scale=1.0,
                          n_frames_before_after=n,
                          crop_coordinates=get_crop_coordinates( 
                              x1=int(event.df.x1.values[frame_index]),
                              y1=int(event.df.y1.values[frame_index]),
                              x2=int(event.df.x2.values[frame_index]),
                              y2=int(event.df.y2.values[frame_index])))
        buffer = vp.get_video(
            save_to_path="{0:03d}_{1}_frames_around_fid_{2}.mp4".format(i, n, list(event.frame_ids)[center_frame_index]))
        i += 1


# debug
def get_center_frame_image(event: Event):
    fp = get_frame_plotter_by_event(event=event, frame_index=get_center_frame_index(event))
    return fp.get_image()


def get_all_images(event: Event):
    images = []

    def add_padding(arr:np.ndarray, frame_ids: List[int]):
        for i in tqdm(range(arr.shape[0])):
            fp = get_frame_plotter(frame_id=frame_ids[i], x1=arr[i,0], y1=arr[i,1], 
                                   x2=arr[i,3], y2=arr[i,4])
            img = fp.get_image()
            # TODO it should be an option to rotate to bee one or bee two, could be used for augmentation
            images.append(crop_image(rotate_image(image=img, bee_orientation=arr[i,2])))

    add_padding(arr=event.frames_before, frame_ids=event.frame_ids_before)

    ids = list(event.frame_ids)
    for i in tqdm(range(len(ids))):
        fp = get_frame_plotter_by_event(event=event, frame_index=i)
        img = fp.get_image()
        # TODO it should be an option to rotate to bee one or bee two, could be used for augmentation
        images.append(crop_image(rotate_image(image=img, bee_orientation=event.df.orient1.values[i])))

    add_padding(arr=event.frames_after, frame_ids=event.frame_ids_after)

    return images


def rotate_image(image: np.ndarray, bee_orientation: float) -> np.ndarray:
    """rotates the image around the center to show one bee at the bottom looking north"""
    global debug_vals_rotation
    debug_vals_rotation.append(math.degrees(bee_orientation))
    return rotate(input=image, angle=math.degrees(bee_orientation), reshape=False)


def crop_image(image: np.ndarray) -> np.ndarray:
    d = (SIZE_BEFORE_ROTATION - FINAL_SIZE) // 2
    return image[d:d + FINAL_SIZE, d:d + FINAL_SIZE]
