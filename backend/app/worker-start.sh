#! /usr/bin/env bash
set -e

python /app/app/celeryworker_pre_start.py

# no events (-E) needed 
celery -A app.worker worker -l info -Q main-queue -c 1 -n backend_worker@%h
