from celery import Celery
from app.core.config import settings

BROKER = f"redis://:{settings.REDIS_PASSWORD}@queue:6379/0"

celery_app = Celery("worker", broker=BROKER)

celery_app.conf.update(
    result_backend = BROKER,
    worker_pool_restarts = True,
    task_routes = {
    # main queue, processed by backend
    # for db operations, emails, basic testing...
    "app.worker.test_celery": "main-queue",
    "app.worker.create_segmentation_from_inference": "main-queue",
    "app.worker.create_model_from_retraining": "main-queue",
    # gpu queue, processed by gpu_worker (have GPU's and are not in containers)
    # for training and inference
    "app.worker.test_pytorch": "gpu-queue",
    "app.worker.infer_unet2d": "gpu-queue",
    "app.worker.train_unet2d": "gpu-queue",
    # TODO conversion queue
    }
)
