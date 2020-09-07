from sqlalchemy.orm import Session

from app import crud, schemas
from app.core.config import settings
from app.db.base import Base  # noqa: F401

# make sure all SQL Alchemy models are imported (app.db.base) before initializing DB
# otherwise, SQL Alchemy might fail to initialize relationships properly
# for more details: https://github.com/tiangolo/full-stack-fastapi-postgresql/issues/28


def init_db(db: Session) -> None:
    # TODO remove in testing
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

    # add main project
    project_in = schemas.ProjectCreate(
            title="Main project",
            description="Main project description",
    )
    main_project = crud.project.create_with_owner(db, obj_in=project_in, owner_id=user.id)

    # add EMBL dataset
    dataset_in = schemas.DatasetCreate(
            title="EMBL Dataset",
            description="EMBL Dataset description",
            location="EM/EMBL/raw/"
    )
    embl_dataset = crud.dataset.create_with_owner(db, obj_in=dataset_in, owner_id=user.id, project_id=main_project.id)

    # add untrained unet2d model
    model_in = schemas.ModelCreate(
            title="Untrained UNet2D",
            description="Untrained UNet2D description",
            location="models/2d/2d.pytorch"
    )
    untrained_unet2d = crud.model.create_with_owner(db, obj_in=model_in, owner_id=user.id, project_id=main_project.id)

    # add unet2d model trained on ground truth
    model_in = schemas.ModelCreate(
            title="Trained UNet2D",
            description="Trained UNet2D description",
            location="models/EMBL/test_run2/best_checkpoint.pytorch"
    )
    trained_unet2d = crud.model.create_with_owner(db, obj_in=model_in, owner_id=user.id, project_id=main_project.id)

    # add ground truth segmentation
    segmentation_in = schemas.SegmentationCreate(
            title="Ground Truth",
            description="EMBL Ground Truth",
            location="segmentations/EMBL/labels/"
    )
    ground_truth = crud.segmentation.create_with_owner(
        db, 
        obj_in=segmentation_in, 
        owner_id=user.id, 
        dataset_id=embl_dataset.id,
        # TODO make model optional
        model_id=untrained_unet2d.id
    )

    # add untrained unet2d segmentation
    segmentation_in = schemas.SegmentationCreate(
            title="Untrained UNet2D segmentation",
            description="Untrained UNet2D segmentation description",
            location="segmentations/EMBL/untrained"
    )
    untrained_segmentation = crud.segmentation.create_with_owner(
        db, 
        obj_in=segmentation_in, 
        owner_id=user.id, 
        dataset_id=embl_dataset.id,
        model_id=untrained_unet2d.id
    )

    # add trained unet2d segmentation
    segmentation_in = schemas.SegmentationCreate(
            title="Trained UNet2D segmentation",
            description="Trained UNet2D segmentation description",
            location="segmentations/EMBL/trained"
    )
    trained_segmentation = crud.segmentation.create_with_owner(
        db, 
        obj_in=segmentation_in, 
        owner_id=user.id, 
        dataset_id=embl_dataset.id,
        model_id=trained_unet2d.id
    )


