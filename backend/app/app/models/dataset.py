from typing import TYPE_CHECKING

from sqlalchemy import Column, ForeignKey, Integer, PickleType, String
from sqlalchemy.orm import relationship

from app.db.base_class import Base

if TYPE_CHECKING:
    from .project import Project  # noqa: F401
    from .user import User  # noqa: F401
    from .annotation import Annotation  # noqa: F401
    from .segmentation import Segmentation  # noqa: F401


class Dataset(Base):
    id = Column(Integer, primary_key=True, index=True)
    title = Column(String, index=True)
    description = Column(String, index=True)
    file_type = Column(String, index=True)
    location = Column(String, index=True)
    resolution = Column(PickleType)
    modality = Column(String, index=True)

    owner_id = Column(Integer, ForeignKey("user.id"))
    owner = relationship("User", back_populates="datasets")

    project_id = Column(Integer, ForeignKey("project.id"))
    project = relationship("Project", back_populates="datasets")

    annotations = relationship("Annotation", back_populates="dataset")
    segmentations = relationship("Segmentation", back_populates="dataset")
