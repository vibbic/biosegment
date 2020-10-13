@rem Note that the Unix version of this script (start_worker.sh) has auto-restart functionality.
@rem On Windows we had to remove this feature since watchmedo auto-restart is broken on Windows (https://github.com/gorakhargosh/watchdog/issues/387)
@echo Please ensure that the ROOT_DATA_FOLDER environment variable is set!
@echo ROOT_DATA_FOLDER=%ROOT_DATA_FOLDER%
celery worker -A app.worker -l info -Q gpu-queue -n gpu_worker@%h
