from sqlalchemy.orm import Session

from app import crud
from app.schemas.model import ModelCreate, ModelUpdate
from app.tests.utils.user import create_random_user
from app.tests.utils.utils import random_lower_string


def test_create_model(db: Session) -> None:
    title = random_lower_string()
    description = random_lower_string()
    location = random_lower_string()
    model_in = ModelCreate(title=title, description=description, location=location)
    user = create_random_user(db)
    model = crud.model.create_with_owner(db=db, obj_in=model_in, owner_id=user.id)
    assert model.title == title
    assert model.description == description
    assert model.owner_id == user.id


def test_get_model(db: Session) -> None:
    title = random_lower_string()
    description = random_lower_string()
    location = random_lower_string()
    model_in = ModelCreate(title=title, description=description, location=location)
    user = create_random_user(db)
    model = crud.model.create_with_owner(db=db, obj_in=model_in, owner_id=user.id)
    stored_model = crud.model.get(db=db, id=model.id)
    assert stored_model
    assert model.id == stored_model.id
    assert model.title == stored_model.title
    assert model.description == stored_model.description
    assert model.owner_id == stored_model.owner_id


def test_update_model(db: Session) -> None:
    title = random_lower_string()
    description = random_lower_string()
    location = random_lower_string()
    model_in = ModelCreate(title=title, description=description, location=location)
    user = create_random_user(db)
    model = crud.model.create_with_owner(db=db, obj_in=model_in, owner_id=user.id)
    description2 = random_lower_string()
    model_update = ModelUpdate(description=description2)
    model2 = crud.model.update(db=db, db_obj=model, obj_in=model_update)
    assert model.id == model2.id
    assert model.title == model2.title
    assert model2.description == description2
    assert model.owner_id == model2.owner_id


def test_delete_model(db: Session) -> None:
    title = random_lower_string()
    description = random_lower_string()
    location = random_lower_string()
    model_in = ModelCreate(title=title, description=description, location=location)
    user = create_random_user(db)
    model = crud.model.create_with_owner(db=db, obj_in=model_in, owner_id=user.id)
    model2 = crud.model.remove(db=db, id=model.id)
    model3 = crud.model.get(db=db, id=model.id)
    assert model3 is None
    assert model2.id == model.id
    assert model2.title == title
    assert model2.description == description
    assert model2.owner_id == user.id
