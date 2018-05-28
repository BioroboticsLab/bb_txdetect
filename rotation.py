import math
from time import time
import numpy as np
from scipy.ndimage.interpolation import rotate

def hide_tag(img, x, y):
    sy, sx = img.shape
    r = 22
    b,a = np.ogrid[-y:sy-y, -x:sx-x]
    mask = a**2 + b**2 <= r**2
    img[mask] = 128
    

def hide_tag_debug(img, x, y, invert):
    sy, sx = img.shape
    r = 22
    b,a = np.ogrid[-y:sy-y, -x:sx-x]
    mask = a**2 + b**2 <= r**2
    img[y-r:y+r,x-r:x+r] = 64 if invert else 128
    img[mask] = 128 if invert else 64
    img[y,x] = 0

    
def center_bee(img, x, y, *args, **kwargs):
    """center one bee for easier rotation"""
    h, w = img.shape
    MIN = 300
    pad_top = max(MIN - y, 0)
    pad_bottom = max(MIN - (h - y), 0)
    pad_left = max(MIN - x, 0)
    pad_right = max(MIN - (w - x), 0)

    if any([pad > 1 for pad in [pad_top, pad_bottom, pad_left, pad_right]]):
        # bee is close to the edge
        # extend image with padding to keep minimum size
        img = np.pad(img, ((pad_top, pad_bottom), (pad_left, pad_right)), 'constant', constant_values=0.5)
        x += pad_left
        y += pad_top
        h, w = img.shape

    if x < w//2:
        minx = 0
        maxx = 2*x
    else:
        minx = 2*x-w
        maxx = w
    if y < h//2:
        miny = 0
        maxy = 2*y
    else:
        miny = 2*y-h
        maxy = h
    try:
        assert maxx - minx >= 2 * MIN -2 and maxy - miny >= 2 * MIN -2
    except AssertionError:
        print(x, y, maxx, minx, maxy, miny, pad_top, pad_bottom, pad_left, pad_right)
        raise
    return img[miny:maxy, minx:maxx]
        


def fix_bee_to_corner(*args, **kwargs):
    """Rotates and crops the image so that one bee is in the bottom left corner 
    and the other is on the top right, if the distance is small enough. 
    If the distance is higher, the 2nd bee will not be visible, but the first 
    is always in the bottom left. Returns the new image."""
    def crop(img, x, y):
        h, w = img.shape
        x = w//2
        y = h//2
        return img[y-128:y, x:x+128]


    def rotate_image(img, x1, y1, x2, y2, invert=False):
        angle = math.degrees(math.atan2(x2-x1, y2-y1))
        angle = -angle + 90 + 45
        return rotate(input=img, angle=angle, reshape=False)

    return _process(*args, **kwargs, rotate_func=rotate_image, crop_func=crop)


def rotation_horizontal(*args, **kwargs):
    """Both bees will have their tags on a horizontal line through the center.
    Returns the new image."""

    def rotate_image(img, x1, y1, x2, y2, *args, **kwargs):
        """rotate to have bees on a horizontal line"""
        angle = math.degrees(math.atan2(x2-x1, y2-y1))
        angle = -angle + 90
        return rotate(input=img, angle=angle, reshape=False)

    return _process(*args, **kwargs, transformations=[center_bee, rotate_image])


def crop_centered_bee_left_edge(img, *args, padding=0, **kwargs):
    """crop around bee to fix it at the left edge"""
    h, w = img.shape
    x = w//2
    y = h//2
    a = 128 + 2*padding
    offs = 22 - padding
    return img[y-a//2:y+a//2, x+offs:x+a+offs]
    

def _process(img, x1, y1, x2, y2, transformations, invert, debug=False):
    if debug:
        def debug_plot():
            import matplotlib.pyplot as plt
            total = len(transformations)
            num_rows = math.ceil(total / 3)
            f, axes = plt.subplots(num_rows, 3, figsize=(18,6*total // 3))
            #import pdb; pdb.set_trace()
            for i in range(total):
                ax = axes[i//3][i % 3] if num_rows > 1 else axes[i]
                title = transformations[i].__doc__
                ax.set_title(title)
                ax.imshow(processed, cmap="gray")
                if i == total - 1:
                    for j in range(i % 3 + 1, 3):
                        axes[-1][j].axis("off")
                    plt.show()
                yield
        plot_gen = debug_plot()
        def plot():
            next(plot_gen)
    else:
        def plot():
            pass
        
    if invert:
        x1, y1, x2, y2 = x2, y2, x1, y1

    #totaltic = tic = time()
    def hide_tags(img, x, y, *args, **kwargs):
        """hide both tags"""
        hide_tag(img=img, x=x1, y=y1)
        hide_tag(img=img, x=x2, y=y2)
        return img
    #print(time() - tic, end="s hide tag; ")
    processed = img
    transformations = [hide_tags, *transformations]
    for transform in transformations:
        #tic = time()
        processed = transform(processed, x1, y1, x2, y2)
        #print(time() - tic, "s", transform.__doc__[:8], end="; ")
        plot()
    #print(time() - totaltic, "_process")
    return processed

