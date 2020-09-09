#!/bin/bash
set -xe

# TODO remove hardcoded conda location
# source ~/miniconda3/etc/profile.d/conda.sh
# TODO detect if celery_neuralnets is installed
# conda activate celery_neuralnets
# TODO add alternative production mode
watchmedo auto-restart --directory=./ --pattern=*.py --recursive -- celery worker -A app.worker -l info -Q gpu-queue -c 2 -n gpu_worker@%h