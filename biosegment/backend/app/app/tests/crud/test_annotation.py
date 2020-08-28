from sqlalchemy.orm import Session

from app import crud
from app.schemas.annotation import AnnotationCreate, AnnotationUpdate
from app.tests.utils.user import create_random_user
from app.tests.utils.utils import random_lower_string


def test_create_annotation(db: Session) -> None:
    title = random_lower_string()
    description = random_lower_string()
    annotation_in = AnnotationCreate(title=title, description=description)
    user = create_random_user(db)
    annotation = crud.annotation.create_with_owner(
        db=db, obj_in=annotation_in, owner_id=user.id
    )
    assert annotation.title == title
    assert annotation.description == description
    assert annotation.owner_id == user.id


def test_get_annotation(db: Session) -> None:
    title = random_lower_string()
    description = random_lower_string()
    annotation_in = AnnotationCreate(title=title, description=description)
    user = create_random_user(db)
    annotation = crud.annotation.create_with_owner(
        db=db, obj_in=annotation_in, owner_id=user.id
    )
    stored_annotation = crud.annotation.get(db=db, id=annotation.id)
    assert stored_annotation
    assert annotation.id == stored_annotation.id
    assert annotation.title == stored_annotation.title
    assert annotation.description == stored_annotation.description
    assert annotation.owner_id == stored_annotation.owner_id


def test_update_annotation(db: Session) -> None:
    title = random_lower_string()
    description = random_lower_string()
    annotation_in = AnnotationCreate(title=title, description=description)
    user = create_random_user(db)
    annotation = crud.annotation.create_with_owner(
        db=db, obj_in=annotation_in, owner_id=user.id
    )
    description2 = random_lower_string()
    annotation_update = AnnotationUpdate(description=description2)
    annotation2 = crud.annotation.update(
        db=db, db_obj=annotation, obj_in=annotation_update
    )
    assert annotation.id == annotation2.id
    assert annotation.title == annotation2.title
    assert annotation2.description == description2
    assert annotation.owner_id == annotation2.owner_id


def test_delete_annotation(db: Session) -> None:
    title = random_lower_string()
    description = random_lower_string()
    annotation_in = AnnotationCreate(title=title, description=description)
    user = create_random_user(db)
    annotation = crud.annotation.create_with_owner(
        db=db, obj_in=annotation_in, owner_id=user.id
    )
    annotation2 = crud.annotation.remove(db=db, id=annotation.id)
    annotation3 = crud.annotation.get(db=db, id=annotation.id)
    assert annotation3 is None
    assert annotation2.id == annotation.id
    assert annotation2.title == title
    assert annotation2.description == description
    assert annotation2.owner_id == user.id
