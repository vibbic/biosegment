import os
import logging

from app.utils import get_folder_list

from PIL import Image
from skimage import io as skio
from app.image_utils import label_to_colors
from app.env import ROOT_DATA_FOLDER

class Dataset:

    def __init__(self, slices_folder):
        assert os.path.exists(slices_folder)
        self.slices_folder = slices_folder
        self.slices = sorted(get_folder_list(slices_folder))
    
    @staticmethod
    def get_models_available():
        return [
            {
                "label": "Untrained UNet2D",
                "value": "models/2d/2d.pytorch"
            },
            {
                "label": "Retrained UNet2D",
                "value": "models/EMBL/test_run2/best_checkpoint.pytorch"
            }
        ]

    @staticmethod
    def get_segmentations_available():
        return [
            {
                "label": "Ground Truth",
                "value": "segmentations/EMBL/labels/"
            },
            {
                "label": "Untrained UNet2D Segmentation",
                "value": "segmentations/EMBL/untrained"
            },
            {
                "label": "Retrained UNet2D Segmentation",
                "value": "segmentations/EMBL/retrained"
            }
        ]

    def get_slice(self, slice_id):
        path = self.slices[slice_id]
        assert os.path.exists(path)
        png = Image.open(path)
        return png

    def get_label(self, labels_folder, slice_id):
        if labels_folder[0] != "/":
            labels_folder = f"{ROOT_DATA_FOLDER}{labels_folder}"
        logging.debug(f"labels_folder: {labels_folder}")
        assert os.path.exists(labels_folder)
        labels = sorted(get_folder_list(labels_folder))
        path = labels[slice_id]
        assert os.path.exists(path)
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

