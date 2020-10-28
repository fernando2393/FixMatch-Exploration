"""
Functions obtained from original Google Research code:
https://github.com/google-research/fixmatch/blob/master/libml/ctaugment.py
"""

from PIL import Image, ImageOps, ImageEnhance, ImageFilter
import numpy as np

BIN_LIST = [[17], [17], [17], [17], [17], [17], [17], [17], [], [8],
            [17, 6], [17], [17], [17], [17], [17], [17], [17], [17]]


# ------- Functions ------- #
def autocontrast(x, level):
    return _imageop(x, ImageOps.autocontrast, level)


def blur(x, level):
    return _filter(x, ImageFilter.BLUR, level)


def brightness(x, brightness_val):
    return _enhance(x, ImageEnhance.Brightness, brightness_val)


def color(x, color_val):
    return _enhance(x, ImageEnhance.Color, color_val)


def contrast(x, contrast_val):
    return _enhance(x, ImageEnhance.Contrast, contrast_val)


def cutout(x, level):
    """Apply cutout to pil_img at the specified level."""
    size = 1 + int(level * min(x.size) * 0.499)
    img_height, img_width = x.size
    height_loc = np.random.randint(low=0, high=img_height)
    width_loc = np.random.randint(low=0, high=img_width)
    upper_coord = (max(0, height_loc - size // 2), max(0, width_loc - size // 2))
    lower_coord = (min(img_height, height_loc + size // 2), min(img_width, width_loc + size // 2))
    pixels = x.load()  # create the pixel map
    for i in range(upper_coord[0], lower_coord[0]):  # for every col:
        for j in range(upper_coord[1], lower_coord[1]):  # For every row
            pixels[i, j] = (127, 127, 127)  # set the color accordingly
    return x


def equalize(x, level):
    return _imageop(x, ImageOps.equalize, level)


def invert(x, level):
    return _imageop(x, ImageOps.invert, level)


def identity(x):
    return x


def posterize(x, level):
    level = 1 + int(level * 7.999)
    return ImageOps.posterize(x, level)


def rescale(x, params):
    s = x.size
    scale = params[0]
    method = params[1]
    scale *= 0.25
    crop = (scale * s[0], scale * s[1], s[0] * (1 - scale), s[1] * (1 - scale))
    methods = (Image.ANTIALIAS, Image.BICUBIC, Image.BILINEAR, Image.BOX, Image.HAMMING, Image.NEAREST)
    method = methods[int(method * 5.99)]
    return x.crop(crop).resize(x.size, method)


def rotate(x, angle):
    angle = int(np.round((2 * angle - 1) * 45))
    return x.rotate(angle)


def sharpness(x, sharpness_val):
    return _enhance(x, ImageEnhance.Sharpness, sharpness_val)


def shear_x(x, shear):
    shear = (2 * shear - 1) * 0.3
    return x.transform(x.size, Image.AFFINE, (1, shear, 0, 0, 1, 0))


def shear_y(x, shear):
    shear = (2 * shear - 1) * 0.3
    return x.transform(x.size, Image.AFFINE, (1, 0, 0, shear, 1, 0))


def smooth(x, level):
    return _filter(x, ImageFilter.SMOOTH, level)


def solarize(x, th):
    th = int(th * 255.999)
    return ImageOps.solarize(x, th)


def translate_x(x, delta):
    delta = (2 * delta - 1) * 0.3
    return x.transform(x.size, Image.AFFINE, (1, 0, delta, 0, 1, 0))


def translate_y(x, delta):
    delta = (2 * delta - 1) * 0.3
    return x.transform(x.size, Image.AFFINE, (1, 0, 0, 0, 1, delta))


# ------- Auxiliary Functions ------- #
def _enhance(x, op, level):
    return op(x).enhance(0.1 + 1.9 * level)


def _imageop(x, op, level):
    return Image.blend(x, op(x), level)


def _filter(x, op, level):
    return Image.blend(x, x.filter(op), level)
