import logging
from json import dump
from pathlib import Path
from typing import Any, List

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session

from app import crud, models, schemas
from app.api import deps

router = APIRouter()


@router.get("/", response_model=List[schemas.Annotation])
def read_annotations(
    db: Session = Depends(deps.get_db),
    skip: int = 0,
    limit: int = 100,
    current_user: models.User = Depends(deps.get_current_active_user),
) -> Any:
    """
    Retrieve annotations.
    """
    if crud.user.is_superuser(current_user):
        annotations = crud.annotation.get_multi(db, skip=skip, limit=limit)
    else:
        annotations = crud.annotation.get_multi_by_owner(
            db=db, owner_id=current_user.id, skip=skip, limit=limit
        )
    return annotations


@router.post("/", response_model=schemas.Annotation)
def create_annotation(
    *,
    db: Session = Depends(deps.get_db),
    annotation_in: schemas.AnnotationCreate,
    current_user: models.User = Depends(deps.get_current_active_user),
) -> Any:
    """
    Create new annotation.
    """
    annotation = crud.annotation.create_with_owner(
        db=db, obj_in=annotation_in, owner_id=current_user.id
    )
    return annotation


@router.put("/{id}", response_model=schemas.Annotation)
def update_annotation(
    *,
    db: Session = Depends(deps.get_db),
    id: int,
    annotation_in: schemas.AnnotationUpdate,
    current_user: models.User = Depends(deps.get_current_active_user),
) -> Any:
    """
    Update an annotation.
    """
    annotation = crud.annotation.get(db=db, id=id)
    if not annotation:
        raise HTTPException(status_code=404, detail="Annotation not found")
    if not crud.user.is_superuser(current_user) and (
        annotation.owner_id != current_user.id
    ):
        raise HTTPException(status_code=400, detail="Not enough permissions")

    if annotation_in.shapes:
        try:
            logging.info(f"location: {annotation.location}")
            assert annotation.location
            annotations_location = Path(f"/data/{annotation.location}")
            # TODO define behaviour if folder exists
            annotations_location.parent.mkdir(parents=True, exist_ok=True)
            annotations_location.touch()
            logging.info(f"location: {annotations_location}")
            with annotations_location.open(mode="w") as fp:
                dump(annotation_in.shapes, fp)
        except Exception as e:
            logging.info(f"Error saving annotations: {e}")
            raise HTTPException(
                status_code=500,
                detail="Something went wrong during saving the annotations",
            )
    annotation = crud.annotation.update(db=db, db_obj=annotation, obj_in=annotation_in)
    return annotation


@router.get("/{id}", response_model=schemas.Annotation)
def read_annotation(
    *,
    db: Session = Depends(deps.get_db),
    id: int,
    current_user: models.User = Depends(deps.get_current_active_user),
) -> Any:
    """
    Get annotation by ID.
    """
    annotation = crud.annotation.get(db=db, id=id)
    if not annotation:
        raise HTTPException(status_code=404, detail="Annotation not found")
    if not crud.user.is_superuser(current_user) and (
        annotation.owner_id != current_user.id
    ):
        raise HTTPException(status_code=400, detail="Not enough permissions")
    return annotation


@router.delete("/{id}", response_model=schemas.Annotation)
def delete_annotation(
    *,
    db: Session = Depends(deps.get_db),
    id: int,
    current_user: models.User = Depends(deps.get_current_active_user),
) -> Any:
    """
    Delete an annotation.
    """
    annotation = crud.annotation.get(db=db, id=id)
    if not annotation:
        raise HTTPException(status_code=404, detail="Annotation not found")
    if not crud.user.is_superuser(current_user) and (
        annotation.owner_id != current_user.id
    ):
        raise HTTPException(status_code=400, detail="Not enough permissions")
    annotation = crud.annotation.remove(db=db, id=id)
    return annotation
