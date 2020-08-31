from app.api import base

ENTRYPOINT_BASE = "datasets"

def get(id, **kwargs):
    return base.get(f"{ENTRYPOINT_BASE}/{id}", **kwargs)

def get_multi(**kwargs):
    return base.get(ENTRYPOINT_BASE, **kwargs)

