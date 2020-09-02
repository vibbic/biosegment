from sqlalchemy.orm import Session

from app import crud
from app.schemas.project import ProjectCreate, ProjectUpdate
from app.tests.utils.user import create_random_user
from app.tests.utils.utils import random_lower_string


def test_create_project(db: Session) -> None:
    title = random_lower_string()
    description = random_lower_string()
    project_in = ProjectCreate(title=title, description=description)
    user = create_random_user(db)
    project = crud.project.create_with_owner(db=db, obj_in=project_in, owner_id=user.id)
    assert project.title == title
    assert project.description == description
    assert project.owner_id == user.id


def test_get_project(db: Session) -> None:
    title = random_lower_string()
    description = random_lower_string()
    project_in = ProjectCreate(title=title, description=description)
    user = create_random_user(db)
    project = crud.project.create_with_owner(db=db, obj_in=project_in, owner_id=user.id)
    stored_project = crud.project.get(db=db, id=project.id)
    assert stored_project
    assert project.id == stored_project.id
    assert project.title == stored_project.title
    assert project.description == stored_project.description
    assert project.owner_id == stored_project.owner_id


def test_update_project(db: Session) -> None:
    title = random_lower_string()
    description = random_lower_string()
    project_in = ProjectCreate(title=title, description=description)
    user = create_random_user(db)
    project = crud.project.create_with_owner(db=db, obj_in=project_in, owner_id=user.id)
    description2 = random_lower_string()
    project_update = ProjectUpdate(description=description2)
    project2 = crud.project.update(db=db, db_obj=project, obj_in=project_update)
    assert project.id == project2.id
    assert project.title == project2.title
    assert project2.description == description2
    assert project.owner_id == project2.owner_id


def test_delete_project(db: Session) -> None:
    title = random_lower_string()
    description = random_lower_string()
    project_in = ProjectCreate(title=title, description=description)
    user = create_random_user(db)
    project = crud.project.create_with_owner(db=db, obj_in=project_in, owner_id=user.id)
    project2 = crud.project.remove(db=db, id=project.id)
    project3 = crud.project.get(db=db, id=project.id)
    assert project3 is None
    assert project2.id == project.id
    assert project2.title == title
    assert project2.description == description
    assert project2.owner_id == user.id
