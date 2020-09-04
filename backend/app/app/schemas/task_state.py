from typing import Optional

from pydantic import BaseModel


class TaskState(BaseModel):
    state: str
    current: Optional[int]
    total: Optional[int]
