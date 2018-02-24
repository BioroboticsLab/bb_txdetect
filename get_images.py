import math
import os
import datetime
from bb_backend.api import FramePlotter
from scipy.ndimage.interpolation import rotate
from scipy.misc import imsave
import numpy as np
from map_data import Observation

FINAL_SIZE = 128
SIZE_BEFORE_ROTATION = math.ceil(FINAL_SIZE * math.sqrt(2))
IMAGE_FOLDER = "images"
TAG_RADIUS = 22


def get_crop_coordinates(x1: int, y1: int, x2: int, y2: int, size: int = SIZE_BEFORE_ROTATION):
    xc = min(x1, x2) + abs(x1 - x2) // 2
    yc = min(y1, y2) + abs(y1 - y2) // 2
    return int(xc - size // 2), int(yc - size // 2), int(xc + size // 2 - 1), int(yc + size // 2 - 1)


def get_frame_plotter(obs: Observation, decode_n_frames: int, crop_coordinates: [int]) -> FramePlotter:
    return FramePlotter(frame_id=int(obs.frame_id), scale=1.0,
                        crop_coordinates=crop_coordinates,
                        raw=True, no_rotate=True,
                        decode_n_frames=decode_n_frames)


def save_images(observations: [Observation], index: int):
    def save(hide_tags: bool):
        if hide_tags:
            x_offset = crop_coordinates[0]
            y_offset = crop_coordinates[1]
            for j in range(2):
                x = obs.xs[j] - x_offset
                y = obs.ys[j] - y_offset
                hide_tag(image=image, x=x, y=y)
        
        for j in range(2):
            if hide_tags:
                subfolder = "{}/{}_hidden_tags".format(folder, j)
            else:
                subfolder = "{}/{}".format(folder, j)
            if not os.path.isdir(subfolder):
                os.mkdir(subfolder)
            
            imsave("{}/{:03}_{}.png".format(subfolder, i, obs.file_name), 
                   crop_image(rotate_image(image, obs.orientations[j])))


    first = observations[0]
    last = observations[-1]
    folder = "{}/{:05}_{}_{}_{}_{}_{}_{}".format(IMAGE_FOLDER,
                                                 index,
                                                 first.frame_id, last.frame_id,
                                                 first.xs[0], last.xs[0],
                                                 first.ys[0], last.ys[0])
    if not os.path.isdir(IMAGE_FOLDER):
        os.mkdir(IMAGE_FOLDER)

    if os.path.isdir(folder):
        raise NameError("folder path for saving images already exists: {}".format(folder))

    os.mkdir(folder)

    for i, obs in enumerate(observations):
        if i == 0:
            decode_n_frames = len(observations)
        else:
            decode_n_frames = None

        crop_coordinates = get_crop_coordinates(obs.xs[0], obs.ys[0], 
                                                obs.xs[1], obs.ys[1])
        image = get_frame_plotter(obs=obs, decode_n_frames=decode_n_frames, 
                                  crop_coordinates=crop_coordinates).get_image()

        save(hide_tags=False)
        save(hide_tags=True)

    print(str(datetime.datetime.now()), "event", index, "complete")
    

def rotate_image(image: np.ndarray, bee_orientation: float) -> np.ndarray:
    """rotates the image around the center to show one bee at the bottom looking north"""
    rotated = rotate(input=image, angle=math.degrees(
        bee_orientation), reshape=False)
    return rotated


def crop_image(image: np.ndarray) -> np.ndarray:
    d = (SIZE_BEFORE_ROTATION - FINAL_SIZE) // 2
    cropped = image[d:d + FINAL_SIZE, d:d + FINAL_SIZE]
    return cropped


def hide_tag(image: np.ndarray, x: int, y: int):
    n = SIZE_BEFORE_ROTATION
    r = TAG_RADIUS
    b,a = np.ogrid[-y:n-y, -x:n-x]
    mask = a*a + b*b <= r*r
    image[mask] = 128
