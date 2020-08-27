from sqlalchemy import Boolean, Column, ForeignKey, Integer, String, Date
from sqlalchemy.orm import relationship

from ..db.base_class import Base

class Project(Base):
    id = Column(Integer, primary_key=True, index=True)
    name = Column(String, unique=True, index=True)
    is_active = Column(Boolean, default=True)
    start = Column(Date, index=True)
    stop = Column(Date, index=True)
    description = Column(String)
    owner_id = Column(Integer, ForeignKey("user.id"))

    owner = relationship("User", back_populates="projects")