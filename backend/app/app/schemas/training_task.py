from typing import List, Optional

from pydantic import BaseModel

from app.schemas.dataset import Resolution

class TrainingTaskBase(BaseModel):
    # optional extra parameters
    # note: duplication with default parameters of task train_unet2d
    orientations: Optional[List[int]] = [
        0,
    ]
    seed: Optional[int] = 0
    device: Optional[int] = 0
    print_stats: Optional[int] = 50
    fm: Optional[int] = 1
    levels: Optional[int] = 4
    dropout: Optional[float] = 0.0
    norm: Optional[str] = "instance"
    activation: Optional[str] = "relu"
    in_channels: Optional[int] = 1
    loss: Optional[str] = "ce"
    lr: Optional[float] = 1e-3
    step_size: Optional[int] = 10
    gamma: Optional[float] = 0.9
    epochs: Optional[int] = 50
    len_epoch: Optional[int] = 100
    test_freq: Optional[int] = 1
    train_batch_size: Optional[int] = 1
    test_batch_size: Optional[int] = 1


class TrainingTask(TrainingTaskBase):
    data_dir: str
    log_dir: str
    resolution: Resolution
    classes_of_interest: List[int]

    # training or retraining
    retrain_model: Optional[str]
