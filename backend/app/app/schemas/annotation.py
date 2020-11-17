from enum import Enum
from typing import Dict, List, Optional

from pydantic import BaseModel

SHAPES = Dict[str, List[Dict]]


class AnnotationFileType(str, Enum):
    json = "json"


class Shapes(BaseModel):
    shapes: Optional[SHAPES] = None


# Shared properties
class AnnotationBase(BaseModel):
    title: Optional[str] = None
    description: Optional[str] = None
    location: Optional[str] = None
    file_type: Optional[AnnotationFileType] = None


# Properties to receive on annotation creation
class AnnotationCreate(AnnotationBase):
    title: str


# Properties to receive on annotation update
class AnnotationUpdate(AnnotationBase):
    shapes: SHAPES = None


# Properties shared by models stored in DB
class AnnotationInDBBase(AnnotationBase):
    id: int
    title: str
    owner_id: int

    class Config:
        orm_mode = True


# Properties to return to client
class Annotation(AnnotationInDBBase):
    pass


# Properties properties stored in DB
class AnnotationInDB(AnnotationInDBBase):
    pass
