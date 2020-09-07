import logging

from app.Dataset import Dataset
from app.env import ROOT_DATA_FOLDER
import app.api as api

class DatasetStore(object):

    __instance = None

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
        datasets = api.dataset.get_multi()
        logging.debug(f"Datasets: {datasets}")
        # task_id = test_celery(json={"timeout": 10})
        return datasets

    @staticmethod
    def get_dataset(dataset_id):
        return Dataset(dataset_id)