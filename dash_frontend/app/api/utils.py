from app.api import base

ENTRYPOINT_BASE = "utils"

def get(id, **kwargs):
    return base.get(f"{ENTRYPOINT_BASE}/{id}", **kwargs)

def get_multi(**kwargs):
    return base.get(ENTRYPOINT_BASE, **kwargs)

def infer(**kwargs):
    return base.post(f"{ENTRYPOINT_BASE}/infer/", **kwargs)