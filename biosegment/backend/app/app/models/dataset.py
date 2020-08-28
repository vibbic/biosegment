from typing import TYPE_CHECKING

from sqlalchemy import Column, ForeignKey, Integer, String
from sqlalchemy.orm import relationship

from app.db.base_class import Base

if TYPE_CHECKING:
    from .project import Project  # noqa: F401


class Dataset(Base):
    id = Column(Integer, primary_key=True, index=True)
    title = Column(String, index=True)
    description = Column(String, index=True)
    file_type = Column(String, index=True)
    location = Column(String, index=True)
    resolution_x = Column(Integer, index=True)
    resolution_y = Column(Integer, index=True)
    resolution_z = Column(Integer, index=True)
    modality = Column(String, index=True)

    owner_id = Column(Integer, ForeignKey("project.id"))
    owner = relationship("Project", back_populates="datasets")
