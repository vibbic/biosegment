from typing import TYPE_CHECKING

from sqlalchemy import Boolean, Column, Integer, String
from sqlalchemy.orm import relationship

from app.db.base_class import Base

if TYPE_CHECKING:
    from .item import Item  # noqa: F401
    from .project import Project  # noqa: F401
    from .model import Model  # noqa: F401
    from .annotation import Annotation  # noqa: F401
    from .segmentation import Segmentation  # noqa: F401
    from .dataset import Dataset  # noqa: F401


class User(Base):
    id = Column(Integer, primary_key=True, index=True)
    full_name = Column(String, index=True)
    email = Column(String, unique=True, index=True, nullable=False)
    hashed_password = Column(String, nullable=False)
    is_active = Column(Boolean(), default=True)
    is_superuser = Column(Boolean(), default=False)

    items = relationship("Item", back_populates="owner")
    annotations = relationship("Annotation", back_populates="owner")
    segmentations = relationship("Segmentation", back_populates="owner")
    datasets = relationship("Dataset", back_populates="owner")
    projects = relationship("Project", back_populates="owner")
    models = relationship("Model", back_populates="owner")
