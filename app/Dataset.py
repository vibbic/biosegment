from utils import get_folder_list

from PIL import Image
from skimage import io as skio
from image_utils import label_to_colors

class Dataset:

    def __init__(self, slices_folder="data/EM/EMBL/raw/", labels_folder="data/EM/EMBL/labels/"):
        self.slices_folder = slices_folder
        self.labels_folder = labels_folder
        self.slices = sorted(get_folder_list(slices_folder))
        self.labels = sorted(get_folder_list(labels_folder))

    def get_slice(self, slice_id):
        path = self.slices[slice_id]
        png = Image.open(path)
        return png

    def get_label(self, slice_id):
        path = self.labels[slice_id]
        image_array = skio.imread(path)
        recolored_image_array = label_to_colors(image_array, **{
            "alpha":[128, 128], 
            "color_class_offset":0,
            "no_map_zero": True
        })
        png = Image.fromarray(recolored_image_array)
        return png

    def get_dimensions(self):
        return {
            "min": 0, 
            "max": len(self.slices)
        }

