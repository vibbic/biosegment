from typing import TYPE_CHECKING

from sqlalchemy import Column, ForeignKey, Integer, String
from sqlalchemy.orm import relationship

from app.db.base_class import Base

if TYPE_CHECKING:
    from .user import User  # noqa: F401
    from .dataset import Dataset  # noqa: F401
    from .model import Model  # noqa: F401
    from .annotation import Annotation  # noqa: F401


class Segmentation(Base):
    id = Column(Integer, primary_key=True, index=True)
    title = Column(String, index=True)
    description = Column(String, index=True)
    file_type = Column(String, index=True)
    location = Column(String, index=True)

    owner_id = Column(Integer, ForeignKey("user.id"))
    owner = relationship("User", back_populates="segmentations")

    dataset_id = Column(Integer, ForeignKey("dataset.id"))
    dataset = relationship("Dataset", back_populates="segmentations")

    model_id = Column(Integer, ForeignKey("model.id"))
    model = relationship("Model", back_populates="segmentations")

    annotations = relationship("Annotation", back_populates="segmentation")
