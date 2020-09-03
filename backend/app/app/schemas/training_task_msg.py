from typing import List, Optional

from pydantic import BaseModel


class TrainingTaskMsg(BaseModel):
    data_dir: str
    log_dir: str
    input_size: List[int]
    classes_of_interest: List[int]

    # training or retraining
    retrain_model: Optional[str]

    # optional extra parameters
    orientations: Optional[List[int]]
    seed: Optional[int]
    device: Optional[int]
    print_stats: Optional[int]
    fm: Optional[int]
    levels: Optional[int]
    dropout: Optional[float]
    norm: Optional[str]
    activation: Optional[str]
    in_channels: Optional[int]
    loss: Optional[str]
    lr: Optional[float]
    step_size: Optional[int]
    gamma: Optional[float]
    epochs: Optional[float]
    len_epoch: Optional[int]
    test_freq: Optional[int]
    train_batch_size: Optional[int]
    trest_batch_size: Optional[int]
