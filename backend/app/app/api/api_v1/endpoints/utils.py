import logging
from typing import Any

from celery.result import AsyncResult
from fastapi import APIRouter, Depends, HTTPException
from pydantic.networks import EmailStr
from sqlalchemy.orm import Session

from app import crud, models, schemas
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
    logging.info(args)
    # TODO check ownership
    # annotation = crud.annotation.get(db=db, id=args.annotation_id)
    # if not annotation:
    #     raise HTTPException(status_code=404, detail="Annotation not found")
    annotation = crud.annotation.get(db=db, id=args.annotation_id)
    if not annotation:
        raise HTTPException(status_code=404, detail="Annotation not found")
    if not annotation.location:
        raise HTTPException(status_code=404, detail="Annotation has no location")
    try:
        dataset = crud.dataset.get(db=db, id=annotation.dataset_id)
        assert dataset
    except:
        raise HTTPException(status_code=404, detail="Dataset not found")
    if not dataset.location:
        raise HTTPException(status_code=404, detail="Dataset has no location")
    if not dataset.resolution:
        raise HTTPException(status_code=404, detail="Dataset has no resolution")
    data_dir = dataset.location
    # TODO write annotations out as pngs
    log_dir = args.location
    retrain_model = args.from_model_id
    # 0 is Falsy
    if retrain_model is not None:
        model = crud.model.get(db=db, id=retrain_model)
        if not model:
            raise HTTPException(status_code=404, detail="Model not found")
        retrain_model_location = model.location
    # remove unused entries like database ids
    kwargs = args.dict(
        exclude={"annotation", "dataset_id", "location", "from_model_id"}
    )
    # add task specific arguments and keep optional training parameters
    kwargs.update(
        {
            "data_dir": data_dir,
            "log_dir": log_dir,
            "annotation_dir": annotation.location,
            "retrain_model": retrain_model_location if retrain_model else None,
            "obj_in": {"title": args.title, "location": args.location},
            "owner_id": current_user.id,
            "project_id": dataset.project_id,
            # TODO use dict for dataset resolution
            "resolution": dataset.resolution,
        }
    )
    task = celery_app.send_task(
        "app.worker.train_unet2d",
        # TODO no hardcoding
        kwargs=kwargs,
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
    dataset = crud.dataset.get(db=db, id=args.dataset_id)
    if not dataset:
        raise HTTPException(status_code=404, detail="Dataset not found")
    data_dir = dataset.location

    model_db = crud.model.get(db=db, id=args.model_id)
    if not model_db:
        raise HTTPException(status_code=404, detail="Model not found")
    model = model_db.location
    write_dir = args.location
    file_type = dataset.file_type
    task = celery_app.send_task(
        "app.worker.infer_unet2d",
        args=[data_dir, model, write_dir, file_type],
        # TODO no hardcoding
        kwargs={
            "obj_in": {
                "title": args.title,
                "location": args.location,
                "file_type": file_type,
            },
            "owner_id": current_user.id,
            "dataset_id": args.dataset_id,
            "model_id": args.model_id,
        },
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
    timeout: int,
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
