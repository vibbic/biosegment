#!/bin/bash

exec 3>&1 4>&2
trap 'exec 2>&4 1>&3' 0 1 2 3
exec 1> worker.log 2>&1
# Everything below will go to the file 'log.out':

set -xe

# get ROOT_DATA_FOLDER from .env
# TODO allow use of .env.prod
export "$(grep "ROOT_DATA_FOLDER" ../.env | xargs)"
export "$(grep "REDIS_PASSWORD" ../.env | xargs)"

BROKER="redis://:${REDIS_PASSWORD}@${1:-biosegment.ugent.be}:6379/0"
RESULT_BACKEND=${BROKER}

# celery 5.0.0
# -P threads is needed for 
# Error celery: daemonic processes are not allowed to have children
if [[ ${2} ]];
then
    watchmedo auto-restart --directory=./ --pattern=*.py --recursive -- celery --app app.worker -b ${BROKER} --result-backend ${RESULT_BACKEND} worker -P threads -l INFO -Q gpu-queue -n gpu_worker@%h -E -c 1
else
    nohup celery --app app.worker -b ${BROKER} --result-backend ${RESULT_BACKEND} worker --autoscale=2,1 --max-tasks-per-child=1 -P threads -l INFO -Q gpu-queue -n gpu_worker@%h -E &
fi
