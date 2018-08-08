import math
import os
import sys
import datetime
from time import sleep
from functools import reduce
from traceback import print_tb
from scipy.ndimage.interpolation import rotate
from scipy.misc import imsave
import numpy as np
from bb_backend.api import FramePlotter
from load_data import Observation

CROP_BORDER = 300
FINAL_SIZE = 128
SIZE_BEFORE_ROTATION = math.ceil(FINAL_SIZE * math.sqrt(2))
IMAGE_FOLDER = "images_v2_y_events"
TAG_RADIUS = 22


def get_crop_coordinates(x1: int, y1: int, x2: int, y2: int):
    return int(min(x1,x2) - CROP_BORDER), int(min(y1,y2) - CROP_BORDER), \
           int(max(x1,x2) + CROP_BORDER), int(max(y1,y2) + CROP_BORDER)


def get_frame_plotter(obs: Observation, decode_n_frames: int, crop_coordinates: [int]) -> FramePlotter:
    return FramePlotter(frame_id=int(obs.frame_id), scale=1.0,
                        crop_coordinates=crop_coordinates,
                        raw=True, no_rotate=True,
                        decode_n_frames=decode_n_frames)


class MetadataEntry(object):
    def __init__(self, index, frame_id, xs, ys, orientations, label, offset):
        self.index = index
        self.frame_id = frame_id
        self.x1 = xs[0]
        self.x2 = xs[1]
        self.y1 = ys[0]
        self.y2 = ys[1]
        self.orientation1 = orientations[0]
        self.orientation2 = orientations[1]
        self.offset_x = offset[0]
        self.offset_y = offset[1]
        self.label = label

    @property
    def csv_row(self):
        return reduce(lambda a,b: str(a) + "," + str(b), 
                      [self.index, self.label, self.frame_id, self.offset_x, self.offset_y, 
                       self.x1, self.y1, self.x2, self.y2, self.orientation1, self.orientation2]) + "\n"

    

def save_images(observations: [Observation], index: int, image_folder=IMAGE_FOLDER):
    folder_label = "y" if any(o.trophallaxis_observed for o in observations) else "n"
    folder = "{}/{:05}_{}".format(image_folder, index, folder_label)

    #print([x.label for x in observations])
    #return

    if not os.path.isdir(image_folder):
        os.mkdir(image_folder)

    if os.path.isdir(folder):
        raise Exception("folder path for saving images already exists: {}".format(folder))

    os.mkdir(folder)

    metadata = "index,label,frame_id,offset_x,offset_y,x1,y1,x2,y2,orientation1,orientation2\n"

    for i, obs in enumerate(observations):
        if i == 0:
            decode_n_frames = len(observations)
        else:
            decode_n_frames = None

        crop_coordinates = get_crop_coordinates(obs.xs[0], obs.ys[0], 
                                                obs.xs[1], obs.ys[1])
        fp = get_frame_plotter(obs=obs, decode_n_frames=decode_n_frames, 
                               crop_coordinates=crop_coordinates)
        done = False
        while not done:
            try:
                image = fp.get_image()
                done = True
            except Exception:
                sleep_time = 60
                print_tb(sys.exc_info()[2])
                print(sys.exc_info()[1])
                print("data:", fp.to_json())
                print(str(datetime.datetime.now()), 
                      "Exception in fp.get_image(), will automatically retry in {} sec".format(sleep_time))
                sleep(sleep_time)
            
        imsave("{}/{:03}_{}.png".format(folder, i, obs.label), image)
        metadata += MetadataEntry(index=i, frame_id=obs.frame_id, xs=obs.xs, ys=obs.ys, 
                                  orientations=obs.orientations, label=obs.label, 
                                  offset=crop_coordinates[:2]).csv_row

    with open(folder + "/metadata.csv", "w") as metadata_csv:
        metadata_csv.write(metadata)
        
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
