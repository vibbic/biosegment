from raven import Client
from celery import Task

from app.core.celery_app import celery_app
from app.core.config import settings
from app.db.session import SessionLocal

from celery.utils.log import get_task_logger

client_sentry = Client(settings.SENTRY_DSN)
logger = get_task_logger(__name__)

class DatabaseTask(Task):
    _db = None

    @property
    def db(self):
        if self._db is None:
            self._db = SessionLocal()
        return self._db


@celery_app.task(bind=True, acks_late=True)
def test_celery(self, timeout: int) -> str:
    from time import sleep

    for i in range(timeout):
        sleep(1)
        self.update_state(state="PROGRESS", meta={"current": i, "total": timeout})
    return f"test task return after timeout of {timeout} seconds"


@celery_app.task(base=DatabaseTask, bind=True, acks_late=True)
def create_segmentation_from_inference(self, obj_in, owner_id):
    from app import crud

    logger.info("obj_in is {obj_in}")
    segmentation = crud.segmentation.create_with_owner(
            db=self.db, obj_in=obj_in, owner_id=owner_id
    )
    return segmentation.id
