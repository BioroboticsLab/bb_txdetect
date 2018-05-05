import math
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

    
def center_bee(img, x, y):
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


def fix_bee_to_left_edge(*args, **kwargs):
    """Rotates and crops the image so that one bee is at the left edge.
    Both bees will have their tags on a horizontal line through the center.
    If the distance is higher, the 2nd bee will not be visible, but the first 
    is always at the left edge left. Returns the new image."""
    def crop(img, x, y):
        h, w = img.shape
        x = w//2
        y = h//2
        return img[y-64:y+64, x+22:x+128+22]


    def rotate_image(img, x1, y1, x2, y2, invert=False):
        angle = math.degrees(math.atan2(x2-x1, y2-y1))
        angle = -angle + 90
        return rotate(input=img, angle=angle, reshape=False)

    return _process(*args, **kwargs, rotate_func=rotate_image, crop_func=crop)

    

def _process(img, x1, y1, x2, y2, invert, rotate_func, crop_func, debug=False):
    if debug:
        def debug_plot():
            import matplotlib.pyplot as plt
            f, ax = plt.subplots(2,2, figsize=(8,8))
            ax[0][0].imshow(img, cmap="gray")
            yield
            ax[0][1].imshow(processed, cmap="gray")
            yield
            ax[1][0].imshow(processed, cmap="gray")
            yield
            ax[1][1].imshow(processed, cmap="gray")
            plt.show()
            yield
        plot_gen = debug_plot()
        def plot():
            next(plot_gen)
    else:
        def plot():
            pass
        
    if invert:
        tmpx = x1
        tmpy = y1
        x1 = x2
        y1 = y2
        x2 = tmpx
        y2 = tmpy

    hide_tag(img=img, x=x1, y=y1)
    hide_tag(img=img, x=x2, y=y2)
    plot()
    processed = center_bee(img, x1,y1)
    plot()
    processed = rotate_func(processed, x1, y1, x2, y2)
    plot()
    processed = crop_func(processed, x1, y1)
    plot()
    return processed
