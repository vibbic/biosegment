from app.Dataset import Dataset
from app.env import ROOT_DATA_FOLDER

EM_FOLDER = f"{ROOT_DATA_FOLDER}EM/"
SEGMENTATION_FOLDER = f"{ROOT_DATA_FOLDER}segmentations/"

class DatasetStore(object):

    __instance = None
    available = [
        {
            "name": "EMBL Raw",
            "slices": f"{EM_FOLDER}EMBL/raw/",
            "labels": f"{EM_FOLDER}EMBL/labels/"
        },
        {
            "name": "EMBL Segmentation",
            "slices": f"{EM_FOLDER}EMBL/raw/",
            "labels": f"{SEGMENTATION_FOLDER}EMBL/"
        },
        {
            "name": "EMBL Test",
            "slices": f"{EM_FOLDER}EMBL/test/",
            "labels": f"{EM_FOLDER}EMBL/test_labels/"
        },
        {
            "name": "EMBL Validation",
            "slices": f"{EM_FOLDER}EMBL/val/",
            "labels": f"{EM_FOLDER}EMBL/val_labels/"
        },
        {
            "name": "EMBL Training",
            "slices": f"{EM_FOLDER}EMBL/train/",
            "labels": f"{EM_FOLDER}EMBL/train_labels/"
        }
    ]

    def __init__(self):
        """ Virtually private constructor. """
        if DatasetStore.__instance != None:
            raise Exception("This class is a singleton!")
        else:
            DatasetStore.__instance = self

    @staticmethod 
    def getInstance():
        """ Static access method. """
        if DatasetStore.__instance == None:
            DatasetStore()
        return DatasetStore.__instance

    @staticmethod
    def get_names_available():
        return [d["name"] for d in DatasetStore.available]

    @staticmethod
    def get_dataset(name):
        metadata = [d for d in DatasetStore.available if d["name"] == name][0]
        return Dataset(
            slices_folder=metadata["slices"], 
            labels_folder=metadata["labels"])