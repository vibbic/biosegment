from celery import Celery

celery_app = Celery("worker", broker="redis://queue:6379/0")

celery_app.conf.result_backend = "redis://queue:6379/0"

celery_app.conf.task_routes = {
    "app.worker.test_celery": "main-queue",
    "app.worker.test_pytorch": "main-queue",
    "app.worker.infer_unet2d": "main-queue",
    "app.worker.train_unet2d": "main-queue",
}
