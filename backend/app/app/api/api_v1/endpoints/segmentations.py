from typing import Any, List, Union
import logging

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session

from app import crud, models, schemas
from app.api import deps
from .utils import infer_unet2d
from app.core.celery_app import celery_app

router = APIRouter()


@router.get("/", response_model=List[schemas.Segmentation])
def read_segmentations(
    db: Session = Depends(deps.get_db),
    skip: int = 0,
    limit: int = 100,
    current_user: models.User = Depends(deps.get_current_active_user),
) -> Any:
    """
    Retrieve segmentations.
    """
    if crud.user.is_superuser(current_user):
        segmentations = crud.segmentation.get_multi(db, skip=skip, limit=limit)
    else:
        segmentations = crud.segmentation.get_multi_by_owner(
            db=db, owner_id=current_user.id, skip=skip, limit=limit
        )
    return segmentations


@router.post("/", response_model=Union[schemas.Segmentation, schemas.Task])
def create_segmentation(
    *,
    db: Session = Depends(deps.get_db),
    segmentation_in: schemas.SegmentationCreateFromModel,
    current_user: models.User = Depends(deps.get_current_active_user),
) -> Any:
    """
    Create new segmentation, possibly from a model.
    """
    logging.info("Creating segmentation")

    # TODO check which part of Union
    try:
        if segmentation_in.model:
            from_model = True
    except:
            from_model = True
    if from_model:
        # TODO better handling of future model location
        logging.info(f"Creating segmentation from model {segmentation_in}")

        kwargs = dict(segmentation_in)
        kwargs["location"] = segmentation_in.write_dir
        task = celery_app.send_task("app.worker.infer_unet2d", args=[], kwargs=kwargs)
        return {"task_id": f"{task}"}
    else:
        segmentation = crud.segmentation.create_with_owner(
            db=db, obj_in=segmentation_in, owner_id=current_user.id
        )
        return segmentation

# @router.post("/", response_model=schemas.Segmentation)
# def create_segmentation_from_model(
#     *,
#     db: Session = Depends(deps.get_db),
#     segmentation_in: schemas.SegmentationCreateFromModel,
#     current_user: models.User = Depends(deps.get_current_active_user),
# ) -> Any:
#     """
#     Create new segmentation from a model.
#     """
#     segmentation = crud.segmentation.create_with_owner(
#         db=db, obj_in=segmentation_in, owner_id=current_user.id
#     )
#     return segmentation

@router.put("/{id}", response_model=schemas.Segmentation)
def update_segmentation(
    *,
    db: Session = Depends(deps.get_db),
    id: int,
    segmentation_in: schemas.SegmentationUpdate,
    current_user: models.User = Depends(deps.get_current_active_user),
) -> Any:
    """
    Update an segmentation.
    """
    segmentation = crud.segmentation.get(db=db, id=id)
    if not segmentation:
        raise HTTPException(status_code=404, detail="Segmentation not found")
    if not crud.user.is_superuser(current_user) and (
        segmentation.owner_id != current_user.id
    ):
        raise HTTPException(status_code=400, detail="Not enough permissions")
    segmentation = crud.segmentation.update(
        db=db, db_obj=segmentation, obj_in=segmentation_in
    )
    return segmentation


@router.get("/{id}", response_model=schemas.Segmentation)
def read_segmentation(
    *,
    db: Session = Depends(deps.get_db),
    id: int,
    current_user: models.User = Depends(deps.get_current_active_user),
) -> Any:
    """
    Get segmentation by ID.
    """
    segmentation = crud.segmentation.get(db=db, id=id)
    if not segmentation:
        raise HTTPException(status_code=404, detail="Segmentation not found")
    if not crud.user.is_superuser(current_user) and (
        segmentation.owner_id != current_user.id
    ):
        raise HTTPException(status_code=400, detail="Not enough permissions")
    return segmentation


@router.delete("/{id}", response_model=schemas.Segmentation)
def delete_segmentation(
    *,
    db: Session = Depends(deps.get_db),
    id: int,
    current_user: models.User = Depends(deps.get_current_active_user),
) -> Any:
    """
    Delete an segmentation.
    """
    segmentation = crud.segmentation.get(db=db, id=id)
    if not segmentation:
        raise HTTPException(status_code=404, detail="Segmentation not found")
    if not crud.user.is_superuser(current_user) and (
        segmentation.owner_id != current_user.id
    ):
        raise HTTPException(status_code=400, detail="Not enough permissions")
    segmentation = crud.segmentation.remove(db=db, id=id)
    return segmentation
