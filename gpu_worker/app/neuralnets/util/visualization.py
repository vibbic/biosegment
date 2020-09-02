import numpy as np
from skimage.measure import label
from skimage.segmentation import mark_boundaries


def overlay(x, y, colors=None, alpha=0.2, boundaries=False):
    """
    Overlay an image with a segmentation map

    :param x: input (grayscale) image
    :param y: label map (all zeros are assumed background)
    :param colors: a list of 3-tuples specifying the colors that correspond to the different labels (randomly chosen if not specified)
    :param alpha: opacity of the label overlay
    :param boundaries: boolean that specifies whether a boundary map should be extracted prior to the overlay
    :return: a color image with the labels overlayed
    """
    y = y.astype('uint8')

    # if colors are not specific, generate random colors
    if colors is None:
        colors = []
        for l in range(np.max(y)):
            colors.append(tuple(np.random.rand(3)))

    # boundary case
    if boundaries:
        seg_img = np.stack((x, x, x), axis=2)
        for l in range(np.max(y)):
            seg_img = mark_boundaries(seg_img, label((y == (l + 1)).astype('uint8')), mode='thick', color=colors[l])

    # segment case
    else:
        seg_img = np.stack((x, x, x), axis=2)
        r = np.zeros_like(x)
        g = np.zeros_like(x)
        b = np.zeros_like(x)
        for l in range(np.max(y)):
            for d in range(3):
                r[y == (l + 1)] = colors[l][0]
                g[y == (l + 1)] = colors[l][1]
                b[y == (l + 1)] = colors[l][2]
        overlay = np.stack((r, g, b), axis=2)
        w1 = np.ones_like(x) * (1 - alpha)
        w1[y == 0] = 1
        w2 = np.ones_like(x) * alpha
        w2[y == 0] = 0
        seg_img = np.stack((w1, w1, w1), axis=2) * seg_img + np.stack((w2, w2, w2), axis=2) * overlay

    return seg_img
