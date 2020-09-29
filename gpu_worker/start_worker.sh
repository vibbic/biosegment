#!/bin/bash
set -xe

# get ROOT_DATA_FOLDER from .env
export "$(grep "ROOT_DATA_FOLDER" ../.env | xargs)"

# TODO remove hardcoded conda location
# source ~/miniconda3/etc/profile.d/conda.sh
# TODO detect if celery_neuralnets is installed
# conda activate celery_neuralnets
# TODO add alternative production mode
# celery 4.4.7
watchmedo auto-restart --directory=./ --pattern=*.py --recursive -- celery worker -A app.worker -l info -Q gpu-queue -n gpu_worker@%h
# celery 5.0.0
# watchmedo auto-restart --directory=./ --pattern=*.py --recursive -- celery --app app.worker worker -l INFO -Q gpu-queue -n gpu_worker@%h -E -c 1