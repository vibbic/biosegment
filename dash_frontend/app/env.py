import os

API_DOMAIN = os.getenv("API_DOMAIN", "localhost")
DEV_MODE = os.getenv("DEV_MODE", True)
# Environment variable not used, docker-compose maps ROOT_DATA_FOLDER with volume to /data
ROOT_DATA_FOLDER = os.getenv("ROOT_DATA_FOLDER", "/data/")
