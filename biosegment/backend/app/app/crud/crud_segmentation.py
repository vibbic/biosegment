from typing import List

from fastapi.encoders import jsonable_encoder
from sqlalchemy.orm import Session

from app.crud.base import CRUDBase
from app.models.segmentation import Segmentation
from app.schemas.segmentation import SegmentationCreate, SegmentationUpdate


class CRUDSegmentation(CRUDBase[Segmentation, SegmentationCreate, SegmentationUpdate]):
    def create_with_owner(
        self, db: Session, *, obj_in: SegmentationCreate, owner_id: int
    ) -> Segmentation:
        obj_in_data = jsonable_encoder(obj_in)
        db_obj = self.model(**obj_in_data, owner_id=owner_id)
        db.add(db_obj)
        db.commit()
        db.refresh(db_obj)
        return db_obj

    def get_multi_by_owner(
        self, db: Session, *, owner_id: int, skip: int = 0, limit: int = 100
    ) -> List[Segmentation]:
        return (
            db.query(self.model)
            .filter(Segmentation.owner_id == owner_id)
            .offset(skip)
            .limit(limit)
            .all()
        )

    def get_multi_by_dataset(
        self, db: Session, *, dataset_id: int, skip: int = 0, limit: int = 100
    ) -> List[Segmentation]:
        return (
            db.query(self.model)
            .filter(Segmentation.dataset_id == dataset_id)
            .offset(skip)
            .limit(limit)
            .all()
        )

    def get_multi_by_model(
        self, db: Session, *, model_id: int, skip: int = 0, limit: int = 100
    ) -> List[Segmentation]:
        return (
            db.query(self.model)
            .filter(Segmentation.model_id == model_id)
            .offset(skip)
            .limit(limit)
            .all()
        )


segmentation = CRUDSegmentation(Segmentation)
