from celery import Celery

celery_app = Celery("worker", broker="redis://localhost:6379/0")
celery_app.conf.result_backend = 'redis://localhost:6379/0'
# configure celery only via backend code
# celery_app.conf.task_routes = {
#     "app.worker.test_pytorch": "main-queue",
#     "app.worker.infer": "main-queue",
# }
