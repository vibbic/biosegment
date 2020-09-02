from sqlalchemy.orm import Session

from app import crud
from app.schemas.dataset import DatasetCreate, DatasetUpdate
from app.tests.utils.project import create_random_project
from app.tests.utils.utils import random_lower_string


def test_create_dataset(db: Session) -> None:
    title = random_lower_string()
    description = random_lower_string()
    dataset_in = DatasetCreate(title=title, description=description)
    project = create_random_project(db)
    dataset = crud.dataset.create_with_owner(
        db=db, obj_in=dataset_in, owner_id=project.id
    )
    assert dataset.title == title
    assert dataset.description == description
    assert dataset.owner_id == project.id


def test_get_dataset(db: Session) -> None:
    title = random_lower_string()
    description = random_lower_string()
    dataset_in = DatasetCreate(title=title, description=description)
    project = create_random_project(db)
    dataset = crud.dataset.create_with_owner(
        db=db, obj_in=dataset_in, owner_id=project.id
    )
    stored_dataset = crud.dataset.get(db=db, id=dataset.id)
    assert stored_dataset
    assert dataset.id == stored_dataset.id
    assert dataset.title == stored_dataset.title
    assert dataset.description == stored_dataset.description
    assert dataset.owner_id == stored_dataset.owner_id


def test_update_dataset(db: Session) -> None:
    title = random_lower_string()
    description = random_lower_string()
    dataset_in = DatasetCreate(title=title, description=description)
    project = create_random_project(db)
    dataset = crud.dataset.create_with_owner(
        db=db, obj_in=dataset_in, owner_id=project.id
    )
    description2 = random_lower_string()
    dataset_update = DatasetUpdate(description=description2)
    dataset2 = crud.dataset.update(db=db, db_obj=dataset, obj_in=dataset_update)
    assert dataset.id == dataset2.id
    assert dataset.title == dataset2.title
    assert dataset2.description == description2
    assert dataset.owner_id == dataset2.owner_id


def test_delete_dataset(db: Session) -> None:
    title = random_lower_string()
    description = random_lower_string()
    dataset_in = DatasetCreate(title=title, description=description)
    project = create_random_project(db)
    dataset = crud.dataset.create_with_owner(
        db=db, obj_in=dataset_in, owner_id=project.id
    )
    dataset2 = crud.dataset.remove(db=db, id=dataset.id)
    dataset3 = crud.dataset.get(db=db, id=dataset.id)
    assert dataset3 is None
    assert dataset2.id == dataset.id
    assert dataset2.title == title
    assert dataset2.description == description
    assert dataset2.owner_id == project.id
