from celery import Celery

celery_app = Celery("worker")

celery_app.autodiscover_tasks(['app.net'])