import logging
from typing import Any, List

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session

from app import crud, models, schemas
from app.api import deps

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


@router.post("/", response_model=schemas.Segmentation)
def create_segmentation(
    *,
    db: Session = Depends(deps.get_db),
    segmentation_in: schemas.SegmentationCreate,
    current_user: models.User = Depends(deps.get_current_active_user),
) -> Any:
    """
    Create new segmentation, possibly from a model.
    """
    logging.info("Creating segmentation")

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
