import matplotlib.pyplot as plt
import bb_backend.api
from bb_backend.api import FramePlotter, VideoPlotter
from typing import List
import matplotlib.pyplot as plt
from scipy.ndimage.interpolation import rotate
from numpy import ndarray
import math
from tqdm import tqdm

FINAL_SIZE = 128
SIZE_BEFORE_ROTATION = math.ceil(FINAL_SIZE * math.sqrt(2))


def get_crop_coordinates(event: 'Event', frame_index: int, size: int = SIZE_BEFORE_ROTATION):
    x1 = int(event.df.x1.values[frame_index])
    y1 = int(event.df.y1.values[frame_index])
    x2 = int(event.df.x2.values[frame_index])
    y2 = int(event.df.y2.values[frame_index])
    xc = min(x1, x2) + abs(x1 - x2) / 2
    yc = min(y1, y2) + abs(y1 - y2) / 2
    return xc - size / 2, yc - size / 2, xc + size / 2 - 1, yc + size / 2 - 1


def get_frame_plotter(event: 'Event', frame_index: int) -> FramePlotter:
    return FramePlotter(frame_id=list(event.frame_ids)[frame_index], scale=1.0,
                        crop_coordinates=get_crop_coordinates(event=event, frame_index=frame_index))


def get_center_frame_index(event: 'Event') -> int:
    return int(event.begin_frame_idx + (event.end_frame_idx - event.begin_frame_idx) / 2)


def get_videos_around_center(events: List['Event']):
    i = 0
    for event in tqdm(events):
        # print("start fp")
        center_frame_index = get_center_frame_index(event)
        center_fp = get_frame_plotter(event=event, frame_index=center_frame_index)
        # print("start vp")
        n = 10
        vp = VideoPlotter(frames=[center_fp],
                          scale=1.0,
                          n_frames_before_after=n,
                          crop_coordinates=get_crop_coordinates(event=event,
                                                                frame_index=center_frame_index))
        buffer = vp.get_video(
            save_to_path="{0:03d}_{1}_frames_around_fid_{2}.mp4".format(i, n, list(event.frame_ids)[center_frame_index]))
        i += 1


def get_center_frame_image(event: 'Event'):
    fp = get_frame_plotter(event=event, frame_index=get_center_frame_index(event))
    return fp.get_image()


def get_all_images(event: 'Event'):
    images = []
    ids = list(event.frame_ids)
    for i in tqdm(range(len(ids))):
        fp = get_frame_plotter(event=event, frame_index=i)
        images.append(fp.get_image())
    return images


def first_bee_rotation(event: 'Event', frame_index: int) -> float:
    return float(event.df.orient1.values[frame_index])


def rotate_image(image: ndarray, event: 'Event', frame_index: int) -> ndarray:
    """rotates the image around the center to show bee one at the bottom looking north"""
    return rotate(input=image,
                  angle=math.degrees(first_bee_rotation(event=event, frame_index=frame_index)),
                  reshape=False)


def crop_image(image: ndarray) -> ndarray:
    d = (SIZE_BEFORE_ROTATION - FINAL_SIZE) // 2
    return image[d:d + FINAL_SIZE, d:d + FINAL_SIZE]
