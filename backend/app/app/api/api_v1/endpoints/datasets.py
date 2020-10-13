from typing import Any, List

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session

from app import crud, models, schemas
from app.api import deps
from app.api.api_v1.endpoints.annotations import check_annotation_location

router = APIRouter()


@router.get("/", response_model=List[schemas.Dataset])
def read_datasets(
    db: Session = Depends(deps.get_db),
    skip: int = 0,
    limit: int = 100,
    current_user: models.User = Depends(deps.get_current_active_user),
) -> Any:
    """
    Retrieve datasets.
    """
    if crud.user.is_superuser(current_user):
        datasets = crud.dataset.get_multi(db, skip=skip, limit=limit)
    else:
        datasets = crud.dataset.get_multi_by_owner(
            db=db, owner_id=current_user.id, skip=skip, limit=limit
        )
    return datasets


@router.post("/", response_model=schemas.Dataset)
def create_dataset(
    *,
    db: Session = Depends(deps.get_db),
    dataset_in: schemas.DatasetCreate,
    current_user: models.User = Depends(deps.get_current_active_user),
) -> Any:
    """
    Create new dataset.
    """
    dataset = crud.dataset.create_with_owner(
        db=db, obj_in=dataset_in, owner_id=current_user.id
    )
    return dataset


@router.put("/{id}", response_model=schemas.Dataset)
def update_dataset(
    *,
    db: Session = Depends(deps.get_db),
    id: int,
    dataset_in: schemas.DatasetUpdate,
    current_user: models.User = Depends(deps.get_current_active_user),
) -> Any:
    """
    Update an dataset.
    """
    dataset = crud.dataset.get(db=db, id=id)
    if not dataset:
        raise HTTPException(status_code=404, detail="Dataset not found")
    if not crud.user.is_superuser(current_user) and (
        dataset.owner_id != current_user.id
    ):
        raise HTTPException(status_code=400, detail="Not enough permissions")
    dataset = crud.dataset.update(db=db, db_obj=dataset, obj_in=dataset_in)
    return dataset


@router.get("/{id}", response_model=schemas.Dataset)
def read_dataset(
    *,
    db: Session = Depends(deps.get_db),
    id: int,
    current_user: models.User = Depends(deps.get_current_active_user),
) -> Any:
    """
    Get dataset by ID.
    """
    dataset = crud.dataset.get(db=db, id=id)
    if not dataset:
        raise HTTPException(status_code=404, detail="Dataset not found")
    if not crud.user.is_superuser(current_user) and (
        dataset.owner_id != current_user.id
    ):
        raise HTTPException(status_code=400, detail="Not enough permissions")
    return dataset


@router.delete("/{id}", response_model=schemas.Dataset)
def delete_dataset(
    *,
    db: Session = Depends(deps.get_db),
    id: int,
    current_user: models.User = Depends(deps.get_current_active_user),
) -> Any:
    """
    Delete an dataset.
    """
    dataset = crud.dataset.get(db=db, id=id)
    if not dataset:
        raise HTTPException(status_code=404, detail="Dataset not found")
    if not crud.user.is_superuser(current_user) and (
        dataset.owner_id != current_user.id
    ):
        raise HTTPException(status_code=400, detail="Not enough permissions")
    dataset = crud.dataset.remove(db=db, id=id)
    return dataset


@router.get("/{id}/annotations", response_model=List[schemas.Annotation])
def read_annotations(
    *,
    db: Session = Depends(deps.get_db),
    id: int,
    skip: int = 0,
    limit: int = 100,
    current_user: models.User = Depends(deps.get_current_active_user),
) -> Any:
    """
    Retrieve annotations.
    """
    dataset = crud.dataset.get(db=db, id=id)
    if not dataset:
        raise HTTPException(status_code=404, detail="Dataset not found")
    if not crud.user.is_superuser(current_user) and (
        dataset.owner_id != current_user.id
    ):
        raise HTTPException(status_code=400, detail="Not enough permissions")
    annotations = crud.annotation.get_multi_by_dataset(
        db=db, dataset_id=dataset.id, skip=skip, limit=limit
    )
    return annotations


@router.post("/{id}/annotations", response_model=schemas.Annotation)
def create_annotation(
    *,
    db: Session = Depends(deps.get_db),
    id: int,
    annotation_in: schemas.AnnotationCreate,
    current_user: models.User = Depends(deps.get_current_active_user),
) -> Any:
    """
    Create new annotation.
    """
    check_annotation_location(annotation_in)
    annotation = crud.annotation.create_with_owner(
        db=db, obj_in=annotation_in, owner_id=current_user.id, dataset_id=id
    )
    return annotation


@router.get("/{id}/segmentations", response_model=List[schemas.Segmentation])
def read_segmentations(
    *,
    db: Session = Depends(deps.get_db),
    id: int,
    skip: int = 0,
    limit: int = 100,
    current_user: models.User = Depends(deps.get_current_active_user),
) -> Any:
    """
    Retrieve segmentations.
    """
    dataset = crud.dataset.get(db=db, id=id)
    if not dataset:
        raise HTTPException(status_code=404, detail="Dataset not found")
    if not crud.user.is_superuser(current_user) and (
        dataset.owner_id != current_user.id
    ):
        raise HTTPException(status_code=400, detail="Not enough permissions")
    segmentations = crud.segmentation.get_multi_by_dataset(
        db=db, dataset_id=dataset.id, skip=skip, limit=limit
    )
    return segmentations


@router.post("/{id}/segmentations", response_model=schemas.Segmentation)
def create_segmentation(
    *,
    db: Session = Depends(deps.get_db),
    id: int,
    segmentation_in: schemas.SegmentationCreate,
    current_user: models.User = Depends(deps.get_current_active_user),
) -> Any:
    """
    Create new segmentation.
    """
    segmentation = crud.segmentation.create_with_owner(
        db=db, obj_in=segmentation_in, owner_id=current_user.id, dataset_id=id
    )
    return segmentation
