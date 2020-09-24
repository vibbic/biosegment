from app.api import base

ENTRYPOINT_BASE = "models"


def get(id, **kwargs):
    return base.get(f"{ENTRYPOINT_BASE}/{id}", **kwargs)


def get_multi(**kwargs):
    return base.get(ENTRYPOINT_BASE, **kwargs)


def get_multi_for_project(project_id, **kwargs):
    return base.get(f"projects/{project_id}/models", **kwargs)
