from datetime import date
from typing import Optional

from pydantic import BaseModel


# Shared properties
class ProjectBase(BaseModel):
    # be optional for e.g. updates
    title: Optional[str] = None
    description: Optional[str] = None
    start_date: Optional[date] = None
    stop_date: Optional[date] = None


# Properties to receive on project creation
class ProjectCreate(ProjectBase):
    # a new project should have at least a title
    title: str


# Properties to receive on project update
class ProjectUpdate(ProjectBase):
    pass


# Properties shared by models stored in DB
class ProjectInDBBase(ProjectBase):
    id: int
    title: str
    owner_id: int

    class Config:
        orm_mode = True


# Properties to return to client
class Project(ProjectInDBBase):
    pass


# Properties properties stored in DB
class ProjectInDB(ProjectInDBBase):
    pass
