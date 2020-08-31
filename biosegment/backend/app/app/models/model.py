from typing import TYPE_CHECKING

from sqlalchemy import Column, ForeignKey, Integer, String
from sqlalchemy.orm import relationship

from app.db.base_class import Base

if TYPE_CHECKING:
    from .user import User  # noqa: F401
    from .project import Project  # noqa: F401


class Model(Base):
    id = Column(Integer, primary_key=True, index=True)
    title = Column(String, index=True)
    description = Column(String, index=True)
    location = Column(String, index=True)

    owner_id = Column(Integer, ForeignKey("user.id"))
    owner = relationship("User", back_populates="models")

    project_id = Column(Integer, ForeignKey("project.id"))
    project = relationship("Project", back_populates="models")

    segmentations = relationship("Segmentation", back_populates="model")
