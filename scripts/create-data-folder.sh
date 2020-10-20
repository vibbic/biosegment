#!/bin/env bash

set -e

# check if data folder not yet present
DATA_FOLDER="${1:-"data"}"

if [ -d "$DATA_FOLDER" ]; then
  echo "Data folder already created at $DATA_FOLDER"
  exit 0
fi

# create data folder
mkdir "$DATA_FOLDER"
cd "$DATA_FOLDER"

# set data folder structure to that of the config file
cat "../scripts/data_folder_list.txt" | xargs mkdir

echo "TODO add EMBL dataset"
# TODO find EMBL dataset download link
# echo "1. Download training sub-volume and groundtruth at https://www.epfl.ch/labs/cvlab/data/data-em/"
# echo "2. move file training.tif to folder data/EM/EPFL/raw/"
# echo "3. move file training_groundtruth.tif to folder data/segmentations/EPFL/labels/"

echo "TODO add EPFL dataset"
echo "1. Download training sub-volume and groundtruth at https://www.epfl.ch/labs/cvlab/data/data-em/"
echo "2. move file training.tif to folder data/EM/EPFL/raw/"
echo "3. move file training_groundtruth.tif to folder data/segmentations/EPFL/labels/"

# TODO change init_db.py in backend