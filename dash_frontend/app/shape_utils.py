import plotly.express as px

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
