from app.api import base

ENTRYPOINT_BASE = "annotations/"


def get(id, **kwargs):
    return base.get(f"{ENTRYPOINT_BASE}{id}", **kwargs)

def get_shapes(id, **kwargs):
    return base.get(f"{ENTRYPOINT_BASE}{id}/shapes/", **kwargs)

def get_multi(**kwargs):
    return base.get(ENTRYPOINT_BASE, **kwargs)


def get_multi_for_dataset(dataset_id, **kwargs):
    return base.get(f"datasets/{dataset_id}/annotations", **kwargs)

def update(id, **kwargs):
    return base.put(f"{ENTRYPOINT_BASE}{id}", **kwargs)

def create(**kwargs):
    return base.post(f"{ENTRYPOINT_BASE}", **kwargs)

def create_for_dataset(dataset_id, **kwargs):
    return base.post(f"datasets/{dataset_id}/annotations", **kwargs)
