from typing import Optional

from pydantic import BaseModel


# Shared properties
class ModelBase(BaseModel):
    # be optional for e.g. updates
    title: Optional[str] = None
    description: Optional[str] = None
    location: Optional[str] = None


# Properties to receive on model creation
class ModelCreate(ModelBase):
    # a new model should have at least a title
    title: str


# Properties to receive on model update
class ModelUpdate(ModelBase):
    pass


# Properties shared by (database!!) models stored in DB
class ModelInDBBase(ModelBase):
    id: int
    title: str
    owner_id: int

    class Config:
        orm_mode = True


# Properties to return to client
class Model(ModelInDBBase):
    pass


# Properties properties stored in DB
class ModelInDB(ModelInDBBase):
    pass
