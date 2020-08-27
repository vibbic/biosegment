from typing import List, Optional

from pydantic import BaseModel

from datetime import date

class ProjectBase(BaseModel):
    name: str
    start: Optional[date] = None
    stop: Optional[date] = None
    description: Optional[str] = None

class ProjectCreate(ProjectBase):
    pass

class ProjectUpdate(ProjectBase):
    pass

class Project(ProjectBase):
    id: int
    is_active: bool
    owner_id: int

    class Config:
        orm_mode = True