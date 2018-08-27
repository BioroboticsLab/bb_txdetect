import os
from glob import glob
from pathlib import Path
import numpy as np
from skimage.io import imread
import matplotlib.pyplot as plt
from scipy.misc import imsave
from tqdm import tqdm
from rotation import rotation_horizontal, crop_centered_bee_left_edge
from path_constants import INVERT, PAD


def _get_metadata(image_folder: str) -> [[dict]]:
    mds = []
    for p in sorted(glob(image_folder + "/*/metadata.csv")):
        md = []
        with open(p) as f:
            for i, row in enumerate(f):
                row = row.split(",")
                if i == 0:
                    continue
                md.append({'offset_x' : int(row[3]),
                           'offset_y': int(row[4]),
                           'x1' : int(row[5]),
                           'y1' : int(row[6]),
                           'x2' : int(row[7]),
                           'y2' : int(row[8])})
        mds.append(md)
    return mds
    

def _get_coordinates(img: np.ndarray, path: str, metadata: [[dict]]):
    row = metadata[_folder_index(path)][_image_index(path)]
    w = img.shape[1]
    h = img.shape[0]
    x1 = row["x1"]
    x2 = row["x2"]
    y1 = row["y1"]
    y2 = row["y2"]
    offset_x = min(int(row["offset_x"]), 4000-w)
    offset_y = min(int(row["offset_y"]), 3000-h)
    x1 -= offset_x
    x2 -= offset_x
    y1 -= offset_y
    y2 -= offset_y
    assert x1 > 0 and y1 > 0 and x2 > 0 and y2 > 0
    assert x1 < 4000 and y1 < 3000 and x2 < 4000 and y2 < 3000
    return x1, y1, x2, y2
    

def _folder_index(path: str):
    return int(path.split("/")[1].split("_")[0])


def _image_index(path: str):
    return int(path.split("/")[2].split("_")[0])

    
def _crop_and_rotate(metadata: [[dict]], padding: int, input_image_folder: str, invert: bool):
    if not Path(input_image_folder).is_dir():
        raise Exception("input image folder not found: {}".format(input_image_folder))

    output_image_folder = "{}{}{}".format(str(Path(input_image_folder)), PAD, padding)

    if invert:
        output_image_folder += INVERT
    for path in tqdm(sorted(glob("{}/*/*.png".format(input_image_folder)))):
        img = imread(path)
        coordinates = _get_coordinates(img=img, path=path, metadata=metadata)
        
        outimg = crop_centered_bee_left_edge(rotation_horizontal(img, *coordinates, invert=True), padding=16)
        outpath = path.replace(input_image_folder, output_image_folder)
        if not os.path.isdir(output_image_folder):
            os.mkdir(output_image_folder)
        subfolder = output_image_folder + "/" + outpath.split("/")[1]
        if not os.path.isdir(subfolder):
            os.mkdir(subfolder)
        imsave(outpath, outimg)
        
    
def preprocess_images(image_folder: str, padding: int=16):
    """crop and rotate all images and save the processed images 
    in two folders with the name {image_folder}_pad{padding}[_invert]
    Args:
        image_folder: input image path
        padding: padding that is left around the image on each side.
                 so e.g. 128x128 with padding 16 leads to 160x160"""
    metadata = _get_metadata(image_folder=image_folder)
    for invert in [True, False]:
        _crop_and_rotate(metadata=metadata, padding=padding, input_image_folder=image_folder, invert=invert)
    
