from typing import List, Optional

from pydantic import BaseModel


class InferTask(BaseModel):
    model: str
    data_dir: str
    labels_dir: str
    write_dir: str
    input_size: List[int]
    classes_of_interest: List[int]

    # optional extra parameters
    orientations: Optional[List[int]] = [0]
    # device: Optional[int]
    in_channels: Optional[int] = 1
    len_epoch: Optional[int] = 100
    test_batch_size: Optional[int] = 1
