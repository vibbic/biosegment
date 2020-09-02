from fastapi import APIRouter

from app.api.api_v1.endpoints import (
    annotations,
    datasets,
    items,
    login,
    models,
    projects,
    segmentations,
    users,
    utils,
)

api_router = APIRouter()
api_router.include_router(login.router, tags=["login"])
api_router.include_router(users.router, prefix="/users", tags=["users"])
api_router.include_router(utils.router, prefix="/utils", tags=["utils"])
api_router.include_router(items.router, prefix="/items", tags=["items"])
api_router.include_router(
    segmentations.router, prefix="/segmentations", tags=["segmentations"]
)
api_router.include_router(models.router, prefix="/models", tags=["models"])
api_router.include_router(projects.router, prefix="/projects", tags=["projects"])
api_router.include_router(datasets.router, prefix="/datasets", tags=["datasets"])
api_router.include_router(
    annotations.router, prefix="/annotations", tags=["annotations"]
)
