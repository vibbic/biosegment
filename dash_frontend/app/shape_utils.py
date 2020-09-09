import logging 

from cairosvg import svg2png
import skimage
import plotly.express as px
import PIL.Image
import io
import numpy as np
import logging

DEFAULT_STROKE_WIDTH = 3  # gives line width of 2^3 = 8
# the number of different classes for labels
NUM_LABEL_CLASSES = 3
class_label_colormap = px.colors.qualitative.Light24
class_labels = list(range(NUM_LABEL_CLASSES))
# we can't have less colors than classes
assert NUM_LABEL_CLASSES <= len(class_label_colormap)

def class_to_color(n):
    return class_label_colormap[n]

def color_to_class(c):
    return class_label_colormap.index(c)

def shape_to_svg_code(shape):
    stroke_width = shape["line"]["width"]
    try:
        interest = color_to_class(shape["line"]["color"])
    except:
        interest = None
    # TODO support more classes of interest
    hexpart = f"0{interest}"
    stroke_color = f"#{hexpart}{hexpart}{hexpart}"
    logging.debug(f"stroke_color: {stroke_color}")
    path = shape["path"]
    return f"""
<path
    stroke="{stroke_color}"
    stroke-width="{stroke_width}"
    d="{path}"
    fill-opacity="0"
/>
"""

def annotations_to_svg_code(annotations, fig=None, width=None, height=None):
    """
    fig is the plotly.py figure which shape resides in (to get width and height)
    and shape is one of the shapes the figure contains.
    """
    # TODO make sure background annotation is on top
    # TODO order of annotations
    if fig is not None:
        # get width and height
        wrange = next(fig.select_xaxes())["range"]
        hrange = next(fig.select_yaxes())["range"]
        width, height = [max(r) - min(r) for r in [wrange, hrange]]
    else:
        if width is None or height is None:
            raise ValueError("If fig is None, you must specify width and height")
    return f"""
<svg
    width="{width}"
    height="{height}"
    viewBox="0 0 {width} {height}"
>
<rect width="100%" height="100%" fill="black" />
{''.join([shape_to_svg_code(a) for a in annotations])}
</svg>
"""

def annotations_to_png(fig=None, annotations=None, width=None, height=None, write_to=None):
    """
    Like svg2png, if write_to is None, returns a bytestring. If it is a path
    to a file it writes to this file and returns None.
    """
    svg_code = annotations_to_svg_code(fig=fig, annotations=annotations, width=width, height=height)
    r = svg2png(bytestring=svg_code, write_to=write_to)
    logging.debug(f"svg2png return {r}")
    return r

# def shape_to_png(fig=None, shape=None, width=None, height=None, write_to=None):
#     """
#     Like svg2png, if write_to is None, returns a bytestring. If it is a path
#     to a file it writes to this file and returns None.
#     """
#     svg_code = shape_to_svg_code(fig=fig, shape=shape, width=width, height=height)
#     r = svg2png(bytestring=svg_code, write_to=write_to)
#     logging.debug(f"svg2png return {r}")
#     return r


def shapes_to_mask(shape_args, shape_layers):
    """
    Returns numpy array (type uint8) with number of rows equal to maximum height
    of all shapes's bounding boxes and number of columns equal to their number
    of rows.
    shape_args is a list of dictionaries whose keys are the parameters to the
    shape_to_png function.
    The mask is taken to be all the pixels that are non-zero in the resulting
    image from rendering the shape.
    shape_layers is either a number or an array
    if a number, all the layers have the same number in the mask
    if an array, must be the same length as shape_args and each entry is an
    integer in [0...255] specifying the layer number. Note that the convention
    is that 0 means no mask, so generally the layer numbers will be non-zero.
    """
    images = []
    for sa in shape_args:
        pngbytes = shape_to_png(**sa)
        images.append(PIL.Image.open(io.BytesIO(pngbytes)))

    mwidth, mheight = [max([im.size[i] for im in images]) for i in range(2)]
    mask = np.zeros((mheight, mwidth), dtype=np.uint8)
    if type(shape_layers) != type(list()):
        layer_numbers = [shape_layers for _ in shape_args]
    else:
        layer_numbers = shape_layers
    imarys = []
    for layer_num, im in zip(layer_numbers, images):
        # layer 0 is reserved for no mask
        imary = skimage.util.img_as_ubyte(np.array(im))
        imary = np.sum(imary, axis=2)
        imary.resize((mheight, mwidth))
        imarys.append(imary)
        mask[imary != 0] = layer_num
    return mask
