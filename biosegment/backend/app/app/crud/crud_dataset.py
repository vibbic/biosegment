from typing import List, Optional

from fastapi.encoders import jsonable_encoder
from sqlalchemy.orm import Session

from app.crud.base import CRUDBase
from app.models.dataset import Dataset
from app.schemas.dataset import DatasetCreate, DatasetUpdate


class CRUDDataset(CRUDBase[Dataset, DatasetCreate, DatasetUpdate]):
    def create_with_owner(
        self,
        db: Session,
        *,
        obj_in: DatasetCreate,
        owner_id: int,
        project_id: Optional[int] = None
    ) -> Dataset:
        obj_in_data = jsonable_encoder(obj_in)
        db_obj = self.model(**obj_in_data, owner_id=owner_id, project_id=project_id)
        db.add(db_obj)
        db.commit()
        db.refresh(db_obj)
        return db_obj

    def get_multi_by_owner(
        self, db: Session, *, owner_id: int, skip: int = 0, limit: int = 100
    ) -> List[Dataset]:
        return (
            db.query(self.model)
            .filter(Dataset.owner_id == owner_id)
            .offset(skip)
            .limit(limit)
            .all()
        )

    def get_multi_by_project(
        self, db: Session, *, project_id: int, skip: int = 0, limit: int = 100
    ) -> List[Dataset]:
        return (
            db.query(self.model)
            .filter(Dataset.project_id == project_id)
            .offset(skip)
            .limit(limit)
            .all()
        )


dataset = CRUDDataset(Dataset)
