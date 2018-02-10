import math
from typing import List, Dict
from bb_backend.api import FramePlotter, VideoPlotter
from scipy.ndimage.interpolation import rotate
import numpy as np
from tqdm import tqdm
from map_data import Event

FINAL_SIZE = 128
SIZE_BEFORE_ROTATION = math.ceil(FINAL_SIZE * math.sqrt(2))


def get_crop_coordinates(x1: int, y1: int, x2: int, y2: int, size: int = SIZE_BEFORE_ROTATION):
    xc = min(x1, x2) + abs(x1 - x2) // 2
    yc = min(y1, y2) + abs(y1 - y2) // 2
    return int(xc - size // 2), int(yc - size // 2), int(xc + size // 2 - 1), int(yc + size // 2 - 1)


def get_frame_plotter(obs: Observation) -> FramePlotter:
    return FramePlotter(frame_id=int(obs.frame_id), scale=1.0,
                        crop_coordinates=get_crop_coordinates(
                            obs.xs[0], obs.ys[0], obs.xs[1], obs.ys[1]),
                        raw=True)


def get_all_images(event: Event):
    images = []

    for obs in tqdm(event.observations):
        img = get_frame_plotter(obs).get_image()
        images.append(crop_image(rotate_image(
            image=img, bee_orientation=obs.orientations[0])))

    return images


def rotate_image(image: np.ndarray, bee_orientation: float) -> np.ndarray:
    """rotates the image around the center to show one bee at the bottom looking north"""
    rotated = rotate(input=image, angle=math.degrees(
        bee_orientation), reshape=False)
    return rotated


def crop_image(image: np.ndarray) -> np.ndarray:
    d = (SIZE_BEFORE_ROTATION - FINAL_SIZE) // 2
    cropped = image[d:d + FINAL_SIZE, d:d + FINAL_SIZE]
    return cropped
