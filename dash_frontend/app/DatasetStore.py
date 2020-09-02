from app.Dataset import Dataset
from app.env import DATA_PREFIX

class DatasetStore(object):

    __instance = None
    available = [
        {
            "name": "EMBL Raw",
            "slices": f"{DATA_PREFIX}EMBL/raw/",
            "labels": f"{DATA_PREFIX}EMBL/labels/"
        },
        {
            "name": "EMBL Test",
            "slices": f"{DATA_PREFIX}EMBL/test/",
            "labels": f"{DATA_PREFIX}EMBL/test_labels/"
        },
        {
            "name": "EMBL Validation",
            "slices": f"{DATA_PREFIX}EMBL/val/",
            "labels": f"{DATA_PREFIX}EMBL/val_labels/"
        },
        {
            "name": "EMBL Training",
            "slices": f"{DATA_PREFIX}EMBL/train/",
            "labels": f"{DATA_PREFIX}EMBL/train_labels/"
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