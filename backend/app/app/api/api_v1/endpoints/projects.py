from typing import Any, List

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session

from app import crud, models, schemas
from app.api import deps

router = APIRouter()


@router.get("/", response_model=List[schemas.Project])
def read_projects(
    db: Session = Depends(deps.get_db),
    skip: int = 0,
    limit: int = 100,
    current_user: models.User = Depends(deps.get_current_active_user),
) -> Any:
    """
    Retrieve projects.
    """
    if crud.user.is_superuser(current_user):
        projects = crud.project.get_multi(db, skip=skip, limit=limit)
    else:
        projects = crud.project.get_multi_by_owner(
            db=db, owner_id=current_user.id, skip=skip, limit=limit
        )
    return projects


@router.post("/", response_model=schemas.Project)
def create_project(
    *,
    db: Session = Depends(deps.get_db),
    project_in: schemas.ProjectCreate,
    current_user: models.User = Depends(deps.get_current_active_user),
) -> Any:
    """
    Create new project.
    """
    project = crud.project.create_with_owner(
        db=db, obj_in=project_in, owner_id=current_user.id
    )
    return project


@router.put("/{id}", response_model=schemas.Project)
def update_project(
    *,
    db: Session = Depends(deps.get_db),
    id: int,
    project_in: schemas.ProjectUpdate,
    current_user: models.User = Depends(deps.get_current_active_user),
) -> Any:
    """
    Update an project.
    """
    project = crud.project.get(db=db, id=id)
    if not project:
        raise HTTPException(status_code=404, detail="Project not found")
    if not crud.user.is_superuser(current_user) and (
        project.owner_id != current_user.id
    ):
        raise HTTPException(status_code=400, detail="Not enough permissions")
    project = crud.project.update(db=db, db_obj=project, obj_in=project_in)
    return project


@router.get("/{id}", response_model=schemas.Project)
def read_project(
    *,
    db: Session = Depends(deps.get_db),
    id: int,
    current_user: models.User = Depends(deps.get_current_active_user),
) -> Any:
    """
    Get project by ID.
    """
    project = crud.project.get(db=db, id=id)
    if not project:
        raise HTTPException(status_code=404, detail="Project not found")
    if not crud.user.is_superuser(current_user) and (
        project.owner_id != current_user.id
    ):
        raise HTTPException(status_code=400, detail="Not enough permissions")
    return project


@router.delete("/{id}", response_model=schemas.Project)
def delete_project(
    *,
    db: Session = Depends(deps.get_db),
    id: int,
    current_user: models.User = Depends(deps.get_current_active_user),
) -> Any:
    """
    Delete an project.
    """
    project = crud.project.get(db=db, id=id)
    if not project:
        raise HTTPException(status_code=404, detail="Project not found")
    if not crud.user.is_superuser(current_user) and (
        project.owner_id != current_user.id
    ):
        raise HTTPException(status_code=400, detail="Not enough permissions")
    project = crud.project.remove(db=db, id=id)
    return project


@router.get("/{id}/datasets", response_model=List[schemas.Dataset])
def read_datasets(
    *,
    db: Session = Depends(deps.get_db),
    id: int,
    skip: int = 0,
    limit: int = 100,
    current_user: models.User = Depends(deps.get_current_active_user),
) -> Any:
    """
    Retrieve datasets for project.
    """
    project = crud.project.get(db=db, id=id)

    if not project:
        raise HTTPException(status_code=404, detail="Project not found")
    if not crud.user.is_superuser(current_user) and (
        project.owner_id != current_user.id
    ):
        raise HTTPException(status_code=400, detail="Not enough permissions")
    datasets = crud.dataset.get_multi_by_project(
        db=db, project_id=project.id, skip=skip, limit=limit
    )
    return datasets


@router.post("/{id}/datasets", response_model=schemas.Dataset)
def create_dataset(
    *,
    db: Session = Depends(deps.get_db),
    id: int,
    dataset_in: schemas.DatasetCreate,
    current_user: models.User = Depends(deps.get_current_active_user),
) -> Any:
    """
    Create new dataset for project.
    """
    dataset = crud.dataset.create_with_owner(
        db=db, obj_in=dataset_in, owner_id=current_user.id, project_id=id
    )
    return dataset


@router.get("/{id}/models", response_model=List[schemas.Model])
def read_models(
    *,
    db: Session = Depends(deps.get_db),
    id: int,
    skip: int = 0,
    limit: int = 100,
    current_user: models.User = Depends(deps.get_current_active_user),
) -> Any:
    """
    Retrieve models for project.
    """
    project = crud.project.get(db=db, id=id)

    if not project:
        raise HTTPException(status_code=404, detail="Project not found")
    if not crud.user.is_superuser(current_user) and (
        project.owner_id != current_user.id
    ):
        raise HTTPException(status_code=400, detail="Not enough permissions")
    datasets = crud.model.get_multi_by_project(
        db=db, project_id=project.id, skip=skip, limit=limit
    )
    return datasets


@router.post("/{id}/models", response_model=schemas.Dataset)
def create_model(
    *,
    db: Session = Depends(deps.get_db),
    id: int,
    model_id: schemas.ModelCreate,
    current_user: models.User = Depends(deps.get_current_active_user),
) -> Any:
    """
    Create new model for project.
    """
    model = crud.model.create_with_owner(
        db=db, obj_in=model_id, owner_id=current_user.id, project_id=id
    )
    return model
