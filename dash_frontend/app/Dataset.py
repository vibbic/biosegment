import logging
import os

import app.api as api
from app.env import ROOT_DATA_FOLDER
from app.image_utils import label_to_colors, create_collection, convert_array_to_png
from app.utils import get_folder_list
from app.FileType import FileType


class Dataset:
    def __init__(self, dataset_id):
        self.dataset_id = dataset_id
        self.dataset = api.dataset.get(dataset_id)
        self.type = FileType(self.dataset["file_type"])
        self.title = self.dataset["title"]
        self.project_id = self.dataset["owner_id"]
        self.segmentations = {}
        self.annotations = {}
        self.models = {}
        self.location = ROOT_DATA_FOLDER / self.dataset['location']
        self.slices = create_collection(self.location, self.type)
        self.labels = {}

    def get_slice(self, slice_id):
        assert self.slices
        return convert_array_to_png(self.slices[slice_id])

    def recolor_image(self, image_array):
        if self.type == FileType.tif3d:
            # presume single TIFF
            recolored_image_array = label_to_colors(
                image_array,
                alpha=[128],
                color_class_offset=1,
                no_map_zero=True,
            )
        elif self.type == FileType.pngseq:
            recolored_image_array = label_to_colors(
                image_array,
                alpha=[128],
                labels_contiguous=True,
                no_map_zero=True,
            )
        else:
            logging.error(f"Unkown filetype for dataset: {self.dataset}")
            return None
        png = convert_array_to_png(recolored_image_array)
        return png

    def get_labels(self, segment_id):
        try:
            if len(self.segmentations) == 0:
                self.get_segmentations_available()
            labels_folder = ROOT_DATA_FOLDER / self.segmentations[segment_id]['location']
        except:
            logging.debug(
                f"No segment_id key {segment_id} in {self.segmentations.keys()}"
            )
            labels_folder = None
        logging.debug(f"labels_folder: {labels_folder}")
        assert labels_folder.is_dir()
        contents = [str(p) for p in labels_folder.iterdir()]
        # TODO support for different segmentation file type
        collection = create_collection(labels_folder, self.type)
        self.labels[segment_id] = collection

    def get_label(self, segment_id, slice_id):
        if segment_id not in self.labels:
            self.get_labels(segment_id)
        image_array = self.labels[segment_id][slice_id]
        return self.recolor_image(image_array)

    def get_models_available(self):
        models = api.model.get_multi_for_project(self.project_id)
        logging.debug(f"Models: {models}")
        self.models = {m["id"]: m for m in models}
        return [{"label": m["title"], "value": m["id"],} for m in models]

    def get_annotations_available(self):
        annotations = api.annotation.get_multi_for_dataset(self.dataset_id)
        logging.debug(f"Annotations: {annotations}")
        self.annotations = {a["id"]: a for a in annotations}
        return [{"label": a["title"], "value": a["id"],} for a in annotations]

    def get_segmentations_available(self):
        segmentations = api.segmentation.get_multi_for_dataset(self.dataset_id)
        logging.debug(f"Segmentations: {segmentations}")
        self.segmentations = {s["id"]: s for s in segmentations}
        return [{"label": m["title"], "value": m["id"],} for m in segmentations]

    def get_title(self):
        return self.title

    def get_dimensions(self):
        return {"min": 0, "max": self.dataset["resolution"]["z"]}
