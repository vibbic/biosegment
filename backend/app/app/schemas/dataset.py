from typing import Optional

from pydantic import BaseModel

class Resolution(BaseModel):
    x: int
    y: int
    z: int

# Shared properties
class DatasetBase(BaseModel):
    # be optional for e.g. updates
    title: Optional[str] = None
    description: Optional[str] = None
    file_type: Optional[str] = None
    location: Optional[str] = None
    resolution: Optional[Resolution] = None
    modality: Optional[str] = None


# Properties to receive on dataset creation
class DatasetCreate(DatasetBase):
    # a new dataset should have at least a title
    title: str


# Properties to receive on dataset update
class DatasetUpdate(DatasetBase):
    pass


# Properties shared by models stored in DB
class DatasetInDBBase(DatasetBase):
    id: int
    title: str
    owner_id: int

    class Config:
        orm_mode = True


# Properties to return to client
class Dataset(DatasetInDBBase):
    pass


# Properties properties stored in DB
class DatasetInDB(DatasetInDBBase):
    pass
