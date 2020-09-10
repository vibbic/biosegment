import logging
from typing import Any, Optional, Union

from celery.result import AsyncResult
from fastapi import APIRouter, Depends
from pydantic.networks import EmailStr
from sqlalchemy.orm import Session

from app import models, schemas, crud
from app.api import deps
from app.core.celery_app import celery_app
from app.utils import send_test_email

router = APIRouter()

logger = logging.getLogger("api")


@router.post("/train/", response_model=schemas.Task, status_code=201)
def train_unet2d(
    args: schemas.ModelCreateFromAnnotation,
    db: Session = Depends(deps.get_db),
    current_user: models.User = Depends(deps.get_current_active_superuser),
) -> Any:
    """
    Train UNet2D model.
    """
    # TODO check ownership
    data_dir = crud.dataset.get(db=db, id=args.dataset_id).location
    log_dir = args.location
    retrain_model = args.from_model_id
    # 0 is Falsy
    if retrain_model is not None:
        retrain_model = crud.model.get(db=db, id=retrain_model)
    task = celery_app.send_task("app.worker.train_unet2d", 
        args=[data_dir, log_dir, retrain_model], 
        # TODO no hardcoding
        kwargs={
        "obj_in": {
                "title": args.title,
                "location": args.location,
            },
            "owner_id": current_user.id,
            "project_id": args.project_id,
        }
    )
    logging.debug(task)
    return {"task_id": f"{task}"}


@router.post("/infer/", response_model=schemas.Task, status_code=201)
def infer_unet2d(
    args: schemas.SegmentationCreateFromModel,
    db: Session = Depends(deps.get_db),
    current_user: models.User = Depends(deps.get_current_active_superuser),
) -> Any:
    """
    Segment using UNet2D model.
    """
    # TODO check ownership
    data_dir = crud.dataset.get(db=db, id=args.dataset_id).location
    model = crud.model.get(db=db, id=args.model_id).location
    write_dir = args.location
    task = celery_app.send_task("app.worker.infer_unet2d", 
        args=[data_dir, model, write_dir], 
        # TODO no hardcoding
        kwargs={
            "obj_in": {
                "title": args.title,
                "location": args.location,
            },
            "owner_id": current_user.id,
            "dataset_id": args.dataset_id,
            "model_id": args.model_id,
        }
    )
    logging.debug(task)
    return {"task_id": f"{task}"}


@router.post("/test-pytorch/", response_model=schemas.Task, status_code=201)
def test_pytorch(
    msg: schemas.Msg,
    current_user: models.User = Depends(deps.get_current_active_superuser),
) -> Any:
    """
    Test Celery worker and Pytorch.
    """
    task = celery_app.send_task("app.worker.test_pytorch", args=[msg.msg])
    logging.debug(task)
    return {"task_id": f"{task}"}


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
    result = AsyncResult(task.task_id, app=celery_app)
    state = result.state
    response = {"state": state}
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
