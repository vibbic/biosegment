import os

API_DOMAIN = os.getenv('API_DOMAIN', "localhost")
DEV_MODE = os.getenv('DEV_MODE', True)
DATA_PREFIX = os.getenv('DATA_PREFIX', "/data/")