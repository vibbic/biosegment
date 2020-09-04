from app.Dataset import Dataset
from app.env import ROOT_DATA_FOLDER

EM_FOLDER = f"{ROOT_DATA_FOLDER}EM/"

class DatasetStore(object):

    __instance = None
    available = [
        {
            "name": "EMBL",
            "slices": f"{EM_FOLDER}EMBL/raw/"
        },
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
        return Dataset(slices_folder=metadata["slices"])