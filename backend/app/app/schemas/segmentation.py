from typing import Optional

from pydantic import BaseModel


# Shared properties
class SegmentationBase(BaseModel):
    title: Optional[str] = None
    description: Optional[str] = None
    file_type: Optional[str] = None
    location: Optional[str] = None


# Properties to receive on segmentation creation
class SegmentationCreate(SegmentationBase):
    title: str


# TODO allow for infer task options
# TODO remove ids
class SegmentationCreateFromModel(SegmentationCreate):
    model_id: int
    dataset_id: int
    location: str


# Properties to receive on segmentation update
class SegmentationUpdate(SegmentationBase):
    pass


# Properties shared by models stored in DB
class SegmentationInDBBase(SegmentationBase):
    id: int
    title: str
    owner_id: int

    class Config:
        orm_mode = True


# Properties to return to client
class Segmentation(SegmentationInDBBase):
    pass


# Properties properties stored in DB
class SegmentationInDB(SegmentationInDBBase):
    pass
