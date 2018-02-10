import math
import os
from bb_backend.api import FramePlotter
from scipy.ndimage.interpolation import rotate
from scipy.misc import imsave
import numpy as np
from tqdm import tqdm
from map_data import Observation

FINAL_SIZE = 128
SIZE_BEFORE_ROTATION = math.ceil(FINAL_SIZE * math.sqrt(2))
IMAGE_FOLDER = "images"


def get_crop_coordinates(x1: int, y1: int, x2: int, y2: int, size: int = SIZE_BEFORE_ROTATION):
    xc = min(x1, x2) + abs(x1 - x2) // 2
    yc = min(y1, y2) + abs(y1 - y2) // 2
    return int(xc - size // 2), int(yc - size // 2), int(xc + size // 2 - 1), int(yc + size // 2 - 1)


def get_frame_plotter(obs: Observation) -> FramePlotter:
    return FramePlotter(frame_id=int(obs.frame_id), scale=1.0,
                        crop_coordinates=get_crop_coordinates(
                            obs.xs[0], obs.ys[0], obs.xs[1], obs.ys[1]),
                        raw=True)


def save_images(observations: [Observation]):
    first = observations[0]
    last = observations[-1]
    path = "{}/{}_{}_{}_{}_{}_{}".format(IMAGE_FOLDER,
                                         first.frame_id, last.frame_id,
                                         first.xs[0], last.xs[0],
                                         first.ys[0], last.ys[0])
    if not os.path.isdir(IMAGE_FOLDER):
        os.mkdir(IMAGE_FOLDER)

    if os.path.isdir(path):
        raise NameError("folder path for saving images already exists: {}".format(path))

    os.mkdir(path)

    for i, obs in enumerate(tqdm(observations)):
        obs.image = get_frame_plotter(obs).get_image()
        imsave("{}/{:03}_{}.png".format(path, i, obs.file_name), obs.image)



def rotate_image(image: np.ndarray, bee_orientation: float) -> np.ndarray:
    """rotates the image around the center to show one bee at the bottom looking north"""
    rotated = rotate(input=image, angle=math.degrees(
        bee_orientation), reshape=False)
    return rotated


def crop_image(image: np.ndarray) -> np.ndarray:
    d = (SIZE_BEFORE_ROTATION - FINAL_SIZE) // 2
    cropped = image[d:d + FINAL_SIZE, d:d + FINAL_SIZE]
    return cropped
