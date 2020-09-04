import logging
import json
from typing import Any, Optional

from fastapi import APIRouter, Depends
from pydantic.networks import EmailStr
from celery.result import AsyncResult

from app import models, schemas
from app.api import deps
from app.core.celery_app import celery_app
from app.utils import send_test_email

router = APIRouter()

logger = logging.getLogger("api")


@router.post("/train/", response_model=schemas.Msg, status_code=201)
def train_unet2d(
    args: schemas.TrainingTask,
    current_user: models.User = Depends(deps.get_current_active_superuser),
) -> Any:
    """
    Train UNet2D model.
    """
    celery_app.send_task("app.worker.train_unet2d", kwargs=dict(args))
    return {"msg": "Word received"}


@router.post("/infer/", response_model=schemas.Msg, status_code=201)
def infer_unet2d(
    args: schemas.InferTask,
    current_user: models.User = Depends(deps.get_current_active_superuser),
) -> Any:
    """
    Segment using UNet2D model.
    """
    celery_app.send_task("app.worker.infer_unet2d", kwargs=dict(args))
    return {"msg": "Word received"}


@router.post("/test-pytorch/", response_model=schemas.Msg, status_code=201)
def test_pytorch(
    msg: schemas.Msg,
    current_user: models.User = Depends(deps.get_current_active_superuser),
) -> Any:
    """
    Test Celery worker and Pytorch.
    """
    celery_app.send_task("app.worker.test_pytorch", args=[msg.msg])
    return {"msg": "Word received"}


@router.post("/test-celery/", response_model=schemas.Task, status_code=201)
def test_celery(
    timeout: Optional[int],
    current_user: models.User = Depends(deps.get_current_active_superuser),
) -> Any:
    """
    Test Celery worker.
    """
    task = celery_app.send_task("app.worker.test_celery", args=[timeout])
    logging.debug(task)
    return {"task_id": f"{task}"}

@router.post("/poll-task/", response_model=schemas.TaskState, status_code=200)
def poll_task(
    task: schemas.Task,
    current_user: models.User = Depends(deps.get_current_active_superuser),
) -> Any:
    """
    Poll Celery task.
    """
    result = AsyncResult(task.task_id,app=celery_app)
    state = result.state
    response = {
        "state": state
    }
    if state == "PROGRESS":
        response["current"] = result.info.get("current")
        response["total"] = result.info.get("total")
    return response

@router.post("/test-email/", response_model=schemas.Msg, status_code=201)
def test_email(
    email_to: EmailStr,
    current_user: models.User = Depends(deps.get_current_active_superuser),
) -> Any:
    """
    Test emails.
    """
    send_test_email(email_to=email_to)
    return {"msg": "Test email sent"}
