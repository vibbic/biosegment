from typing import Dict, List, Optional

from pydantic import BaseModel


# Shared properties
class AnnotationBase(BaseModel):
    title: Optional[str] = None
    description: Optional[str] = None
    location: Optional[str] = None
    file_type: Optional[str] = None


# Properties to receive on annotation creation
class AnnotationCreate(AnnotationBase):
    title: str


# Properties to receive on annotation update
class AnnotationUpdate(AnnotationBase):
    shapes: Optional[Dict[str, List[Dict]]] = None


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
