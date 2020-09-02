conda activate celery_neuralnets
celery worker -A app.worker -l info -Q main-queue -c 1