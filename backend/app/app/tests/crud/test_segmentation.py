from sqlalchemy.orm import Session

from app import crud
from app.schemas.segmentation import SegmentationCreate, SegmentationUpdate
from app.tests.utils.user import create_random_user
from app.tests.utils.utils import random_lower_string


def test_create_segmentation(db: Session) -> None:
    title = random_lower_string()
    description = random_lower_string()
    segmentation_in = SegmentationCreate(title=title, description=description)
    user = create_random_user(db)
    segmentation = crud.segmentation.create_with_owner(
        db=db, obj_in=segmentation_in, owner_id=user.id
    )
    assert segmentation.title == title
    assert segmentation.description == description
    assert segmentation.owner_id == user.id


def test_get_segmentation(db: Session) -> None:
    title = random_lower_string()
    description = random_lower_string()
    segmentation_in = SegmentationCreate(title=title, description=description)
    user = create_random_user(db)
    segmentation = crud.segmentation.create_with_owner(
        db=db, obj_in=segmentation_in, owner_id=user.id
    )
    stored_segmentation = crud.segmentation.get(db=db, id=segmentation.id)
    assert stored_segmentation
    assert segmentation.id == stored_segmentation.id
    assert segmentation.title == stored_segmentation.title
    assert segmentation.description == stored_segmentation.description
    assert segmentation.owner_id == stored_segmentation.owner_id


def test_update_segmentation(db: Session) -> None:
    title = random_lower_string()
    description = random_lower_string()
    segmentation_in = SegmentationCreate(title=title, description=description)
    user = create_random_user(db)
    segmentation = crud.segmentation.create_with_owner(
        db=db, obj_in=segmentation_in, owner_id=user.id
    )
    description2 = random_lower_string()
    segmentation_update = SegmentationUpdate(description=description2)
    segmentation2 = crud.segmentation.update(
        db=db, db_obj=segmentation, obj_in=segmentation_update
    )
    assert segmentation.id == segmentation2.id
    assert segmentation.title == segmentation2.title
    assert segmentation2.description == description2
    assert segmentation.owner_id == segmentation2.owner_id


def test_delete_segmentation(db: Session) -> None:
    title = random_lower_string()
    description = random_lower_string()
    segmentation_in = SegmentationCreate(title=title, description=description)
    user = create_random_user(db)
    segmentation = crud.segmentation.create_with_owner(
        db=db, obj_in=segmentation_in, owner_id=user.id
    )
    segmentation2 = crud.segmentation.remove(db=db, id=segmentation.id)
    segmentation3 = crud.segmentation.get(db=db, id=segmentation.id)
    assert segmentation3 is None
    assert segmentation2.id == segmentation.id
    assert segmentation2.title == title
    assert segmentation2.description == description
    assert segmentation2.owner_id == user.id
