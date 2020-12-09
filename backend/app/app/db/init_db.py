import json
import logging
from pathlib import Path
from typing import Dict, Tuple

from sqlalchemy.orm import Session

from app import models  # noqa: F401
from app import crud, schemas
from app.core.config import settings
from app.db.base import Base  # noqa: F401
from app.models.dataset import Dataset
from app.schemas.dataset import Resolution

DATA_FOLDER = Path("/data")


def add_dataset(
    db: Session,
    user: models.User,
    project: models.Project,
    model: models.Model,
    name: str,
    resolution: schemas.Resolution,
    file_type: str,
) -> Tuple[models.Dataset, models.Segmentation]:
    title = name
    dataset_in = schemas.DatasetCreate(
        title=title,
        description=f"{title} description",
        location=f"EM/{name}/raw/",
        resolution=resolution,
        file_type=file_type,
    )
    dataset = crud.dataset.create_with_owner(
        db, obj_in=dataset_in, owner_id=user.id, project_id=project.id
    )

    segmentation_in = schemas.SegmentationCreate(
        title="Ground Truth",
        description=f"{name} Ground Truth",
        location=f"segmentations/{name}/labels/",
    )
    ground_truth = crud.segmentation.create_with_owner(
        db,
        obj_in=segmentation_in,
        owner_id=user.id,
        dataset_id=dataset.id,
        # TODO make model optional
        model_id=model.id,
    )
    return dataset, ground_truth


def init_using_setup_config(db: Session, user: models.User, setup: Dict) -> None:
    projects = {}
    models = {}
    datasets: Dict[str, Dataset] = {}

    for p in setup["projects"]:
        title = p["title"]
        project_in = schemas.ProjectCreate(
            title=title, description=f"{title} description",
        )
        db_project = crud.project.create_with_owner(
            db, obj_in=project_in, owner_id=user.id
        )
        projects[title] = db_project
    for m in setup["models"]:
        title = m["title"]
        location = m["location"]
        project = projects[m["project"]]
        model_in = schemas.ModelCreate(
            title=title, description=f"{title} description", location=location,
        )
        db_model = crud.model.create_with_owner(
            db, obj_in=model_in, owner_id=user.id, project_id=project.id
        )
        models[title] = db_model
    for d in setup["datasets"]:
        title = d["title"]
        location = "datasets/{title}/raw"
        file_type = d["file_type"]
        resolution = d["resolution"]
        project = projects[d["project"]]
        # TODO make optional
        model = list(models.values())[0]
        db_dataset, _ = add_dataset(
            db,
            user,
            project,
            model,
            title,
            Resolution(x=resolution[0], y=resolution[1], z=resolution[2],),
            file_type,
        )
        datasets[title] = db_dataset
    for s in setup["segmentations"]:
        title = s["title"]
        location = s["location"]
        dataset = datasets[s["dataset"]]
        model = models[s["model"]]

        segmentation_in = schemas.SegmentationCreate(
            title=title, description=f"{title} description", location=location,
        )
        crud.segmentation.create_with_owner(
            db,
            obj_in=segmentation_in,
            owner_id=user.id,
            dataset_id=dataset.id,
            model_id=model.id,
        )
    for a in setup["annotations"]:
        title = a["title"]
        location = a["location"]
        dataset = datasets[a["dataset"]]

        annotation_in = schemas.AnnotationCreate(
            title=title, description=f"{title} description", location=location,
        )
        crud.annotation.create_with_owner(
            db, obj_in=annotation_in, owner_id=user.id, dataset_id=dataset.id,
        )


# make sure all SQL Alchemy models are imported (app.db.base) before initializing DB
# otherwise, SQL Alchemy might fail to initialize relationships properly
# for more details: https://github.com/tiangolo/full-stack-fastapi-postgresql/issues/28


def init_db(db: Session) -> None:
    # TODO remove in production
    # clean database of testing data
    Base.metadata.drop_all(bind=db.get_bind())

    # Tables should be created with Alembic migrations
    # But if you don't want to use migrations, create
    # the tables un-commenting the next line
    Base.metadata.create_all(bind=db.get_bind())

    user = crud.user.get_by_email(db, email=settings.FIRST_SUPERUSER)
    if not user:
        user_in = schemas.UserCreate(
            email=settings.FIRST_SUPERUSER,
            password=settings.FIRST_SUPERUSER_PASSWORD,
            is_superuser=True,
        )
        user = crud.user.create(db, obj_in=user_in)  # noqa: F841

    setup_file = DATA_FOLDER / "setup.json"

    if setup_file.exists():
        try:
            with setup_file.open() as fh:
                setup = json.load(fh)
            init_using_setup_config(db, user, setup)
        except Exception as e:
            logging.error(f"Init db from setup file {setup_file} failed: {e}")
    else:
        logging.info(f"No setup file detection at {setup_file}")
