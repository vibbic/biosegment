import os
import logging

from app.utils import get_folder_list

from PIL import Image
from skimage import io as skio
from app.image_utils import label_to_colors
from app.env import ROOT_DATA_FOLDER
import app.api as api 

class Dataset:

    def __init__(self, dataset_id):
        self.dataset_id = dataset_id
        dataset = api.dataset.get(dataset_id)
        self.title = dataset["title"]
        slices_folder=f"{ROOT_DATA_FOLDER}{dataset['location']}"
        self.project_id = dataset['owner_id']
        logging.debug(f"Slices folder: {slices_folder}")
        assert os.path.exists(slices_folder)
        self.slices_folder = slices_folder
        self.slices = sorted(get_folder_list(slices_folder))
    
    def get_models_available(self):
        models = api.model.get_multi_for_project(self.project_id)
        logging.debug(f"Models: {models}")
        return [{
            "label": m["title"],
            "value": m["location"],
        } for m in models]

    def get_segmentations_available(self):
        segmentations = api.segmentation.get_multi_for_dataset(self.dataset_id)
        logging.debug(f"Segmentations: {segmentations}")
        return [{
            "label": m["title"],
            "value": m["location"],
        } for m in segmentations]

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

    def get_title(self):
        return self.title

    def get_dimensions(self):
        return {
            "min": 0, 
            "max": len(self.slices)
        }

