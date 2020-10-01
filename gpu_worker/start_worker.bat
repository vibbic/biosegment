@rem TODO Add watchmedo functionality like in start_worker.sh
@echo Please ensure that the ROOT_DATA_FOLDER environment variable is set!
@echo ROOT_DATA_FOLDER=%ROOT_DATA_FOLDER%
celery worker -A app.worker -l info -Q gpu-queue -n gpu_worker@%h
