from app.api import base

ENTRYPOINT_BASE = "utils"

def get(id, **kwargs):
    return base.get(f"{ENTRYPOINT_BASE}/{id}", **kwargs)

def get_multi(**kwargs):
    return base.get(ENTRYPOINT_BASE, **kwargs)

def train(**kwargs):
    return base.post(f"{ENTRYPOINT_BASE}/train/", **kwargs)

def infer(**kwargs):
    return base.post(f"{ENTRYPOINT_BASE}/infer/", **kwargs)

def poll_task(**kwargs):
    return base.post(f"{ENTRYPOINT_BASE}/poll-task/", **kwargs)

def test_celery(json={"timeout": 10}, **kwargs):
    try:
        timeout = json["timeout"]
    except:
        # TODO
        timeout = 10
    return base.post(f"{ENTRYPOINT_BASE}/test-celery/?timeout={timeout}", **kwargs)