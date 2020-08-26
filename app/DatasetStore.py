from Dataset import Dataset

class DatasetStore(object):

    __instance = None
    available = [
        {
            "name": "EMBL Raw",
            "slices": "data/EM/EMBL/raw/",
            "labels": "data/EM/EMBL/labels/"
        },
        {
            "name": "EMBL Test",
            "slices": "data/EM/EMBL/test/",
            "labels": "data/EM/EMBL/test_labels/"
        },
        {
            "name": "EMBL Validation",
            "slices": "data/EM/EMBL/val/",
            "labels": "data/EM/EMBL/val_labels/"
        },
        {
            "name": "EMBL Training",
            "slices": "data/EM/EMBL/train/",
            "labels": "data/EM/EMBL/train_labels/"
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