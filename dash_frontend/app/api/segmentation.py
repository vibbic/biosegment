from app.api import base

ENTRYPOINT_BASE = "segmentations"

def get(id, **kwargs):
    return base.get(f"{ENTRYPOINT_BASE}/{id}", **kwargs)

def get_multi(**kwargs):
    return base.get(ENTRYPOINT_BASE, **kwargs)

def get_multi_for_dataset(dataset_id, **kwargs):
    return base.get(f"datasets/{dataset_id}/segmentations", **kwargs)

def create(**kwargs):
    return base.post(f"{ENTRYPOINT_BASE}", **kwargs)