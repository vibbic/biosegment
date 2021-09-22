#!/bin/bash
set -xe

# get ROOT_DATA_FOLDER from .env
# TODO allow use of .env.prod
export "$(grep "ROOT_DATA_FOLDER" ../.env | xargs)"

BROKER="redis://${1:-localhost}:6379/0"
RESULT_BACKEND=${BROKER}

# TODO remove hardcoded conda location
# source ~/miniconda3/etc/profile.d/conda.sh
# TODO detect if celery_neuralnets is installed
# conda activate celery_neuralnets
# TODO add alternative production mode
# celery 5.0.0
# -P threads is needed for 
# Error celery: daemonic processes are not allowed to have children
watchmedo auto-restart --directory=./ --pattern=*.py --recursive -- celery --app app.worker -b ${BROKER} --result-backend ${RESULT_BACKEND} worker -P threads -l INFO -Q gpu-queue -n gpu_worker@%h -E -c 1