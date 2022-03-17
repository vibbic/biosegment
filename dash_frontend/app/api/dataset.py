from app.api import base

ENTRYPOINT_BASE = "datasets/"


def get(id: int, **kwargs):
    return base.get(f"{ENTRYPOINT_BASE}{id}", **kwargs)


def get_multi(**kwargs):
    return base.get(ENTRYPOINT_BASE)


def get_multi_for_project(project_id: int, **kwargs):
    return base.get(f"projects/{project_id}/datasets", **kwargs)


def create_for_project(project_id: int, **kwargs):
    return base.post(f"projects/{project_id}/datasets", **kwargs)
