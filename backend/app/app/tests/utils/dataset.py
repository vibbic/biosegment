from typing import Optional

from sqlalchemy.orm import Session

from app import crud, models
from app.schemas.dataset import DatasetCreate
from app.tests.utils.project import create_random_project
from app.tests.utils.user import create_random_user
from app.tests.utils.utils import random_lower_string


def create_random_dataset(
    db: Session, *, owner_id: Optional[int] = None, project_id: Optional[int] = None
) -> models.Dataset:
    if owner_id is None:
        user = create_random_user(db)
        owner_id = user.id
    if project_id is None:
        project = create_random_project(db)
        project_id = project.id
    title = random_lower_string()
    description = random_lower_string()
    dataset_in = DatasetCreate(title=title, description=description, id=id)
    return crud.dataset.create_with_owner(
        db=db, obj_in=dataset_in, owner_id=owner_id, project_id=project_id
    )
