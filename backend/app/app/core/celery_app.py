from celery import Celery

celery_app = Celery("worker", broker="amqp://guest@queue//")

celery_app.conf.task_routes = {
    "app.worker.test_celery": "main-queue",
    "app.worker.test_pytorch": "main-queue",
    "app.worker.infer_unet2d": "main-queue",
    "app.worker.train_unet2d": "main-queue",
}
