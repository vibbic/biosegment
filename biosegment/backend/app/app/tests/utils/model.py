from typing import Optional

from sqlalchemy.orm import Session

from app import crud, models
from app.schemas.model import ModelCreate
from app.tests.utils.user import create_random_user
from app.tests.utils.utils import random_lower_string


def create_random_model(
    db: Session, *, owner_id: Optional[int] = None
) -> models.Model:
    if owner_id is None:
        user = create_random_user(db)
        owner_id = user.id
    title = random_lower_string()
    description = random_lower_string()
    location = random_lower_string()
    model_in = ModelCreate(title=title, description=description, location=location, id=id)
    return crud.model.create_with_owner(db=db, obj_in=model_in, owner_id=owner_id)
