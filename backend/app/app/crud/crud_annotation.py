from typing import List, Optional
import json

from fastapi.encoders import jsonable_encoder
from sqlalchemy.orm import Session

from app import ROOT_DATA_FOLDER
from app.crud.base import CRUDBase
from app.models.annotation import Annotation
from app.schemas.annotation import AnnotationCreate, AnnotationUpdate


class CRUDAnnotation(CRUDBase[Annotation, AnnotationCreate, AnnotationUpdate]):
    def create_with_owner(
        self,
        db: Session,
        *,
        obj_in: AnnotationCreate,
        owner_id: int,
        dataset_id: Optional[int] = None,
        segmentation_id: Optional[int] = None
    ) -> Annotation:
        obj_in_data = jsonable_encoder(obj_in)
        db_obj = self.model(
            **obj_in_data,
            owner_id=owner_id,
            dataset_id=dataset_id,
            segmentation_id=segmentation_id
        )
        db.add(db_obj)
        db.commit()
        db.refresh(db_obj)
        return db_obj
    
    def get_multi_by_owner(
        self, db: Session, *, owner_id: int, skip: int = 0, limit: int = 100
    ) -> List[Annotation]:
        return (
            db.query(self.model)
            .filter(Annotation.owner_id == owner_id)
            .offset(skip)
            .limit(limit)
            .all()
        )

    def get_multi_by_dataset(
        self, db: Session, *, dataset_id: int, skip: int = 0, limit: int = 100
    ) -> List[Annotation]:
        return (
            db.query(self.model)
            .filter(Annotation.dataset_id == dataset_id)
            .offset(skip)
            .limit(limit)
            .all()
        )

    def get_multi_by_segmentation(
        self, db: Session, *, segmentation_id: int, skip: int = 0, limit: int = 100
    ) -> List[Annotation]:
        return (
            db.query(self.model)
            .filter(Annotation.segmentation_id == segmentation_id)
            .offset(skip)
            .limit(limit)
            .all()
        )


annotation = CRUDAnnotation(Annotation)
