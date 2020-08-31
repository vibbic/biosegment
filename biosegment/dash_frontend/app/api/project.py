from app.api.base import Base

ENTRYPOINT_BASE = "projects"

def get(id):
    return Base.get(f"{ENTRYPOINT_BASE}/{id}")

def get_multi():
    return Base.get(ENTRYPOINT_BASE)

