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

# TODO add EMBL raw to data/EM/EMBL/raw
# TODO add EMBL labels to /data/segmentations/EMBL/labels