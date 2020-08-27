from typing import List, Optional

from pydantic import BaseModel

from datetime import date


class ItemBase(BaseModel):
    title: str
    description: Optional[str] = None


class ItemCreate(ItemBase):
    pass


class Item(ItemBase):
    id: int
    owner_id: int

    class Config:
        orm_mode = True


class UserBase(BaseModel):
    email: str
    name: str


class UserCreate(UserBase):
    password: str


class User(UserBase):
    id: int
    is_active: bool
    items: List[Item] = []

    class Config:
        orm_mode = True


class ProjectBase(BaseModel):
    name: str
    start: Optional[date] = None
    stop: Optional[date] = None
    description: Optional[str] = None


class ProjectCreate(ProjectBase):
    pass


class Project(ProjectBase):
    id: int
    is_active: bool
    owner: User

    class Config:
        orm_mode = True