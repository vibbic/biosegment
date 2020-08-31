from typing import TYPE_CHECKING

from sqlalchemy import Column, ForeignKey, Integer, String
from sqlalchemy.orm import relationship

from app.db.base_class import Base

if TYPE_CHECKING:
    from .user import User  # noqa: F401
    from .dataset import Dataset  # noqa: F401
    from .segmentation import Segmentation  # noqa: F401


class Annotation(Base):
    id = Column(Integer, primary_key=True, index=True)
    title = Column(String, index=True)
    description = Column(String, index=True)
    location = Column(String, index=True)
    file_type = Column(String, index=True)

    owner_id = Column(Integer, ForeignKey("user.id"))
    owner = relationship("User", back_populates="annotations")

    dataset_id = Column(Integer, ForeignKey("dataset.id"))
    dataset = relationship("Dataset", back_populates="annotations")

    segmentation_id = Column(Integer, ForeignKey("segmentation.id"))
    segmentation = relationship("Segmentation", back_populates="annotations")
