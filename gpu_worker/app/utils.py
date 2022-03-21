import os
from pathlib import Path
import json
import numpy as np

try:
   # expect to be in gpu_worker/ and data/ is located at ..
   os.chdir(Path(".."))
   # but allow for absolute ROOT_DATA_FOLDER
   ROOT_DATA_FOLDER = Path(os.environ["ROOT_DATA_FOLDER"]).resolve()
   print(f"Root data folder {ROOT_DATA_FOLDER}")
except KeyError: 
   import sys
   print("Please set the environment variable ROOT_DATA_FOLDER in .env")
   sys.exit(1)

def create_meta(current, total):
    return {
        "current": current,
        "total": total
    }

def arr_to_str(arr):
    return str(arr.tolist())

def str_to_arr(s):
    arr = json.loads(s)
    arr = np.asarray(arr)
    return arr