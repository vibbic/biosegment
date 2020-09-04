from raven import Client

from app.core.celery_app import celery_app
from app.core.config import settings

client_sentry = Client(settings.SENTRY_DSN)


@celery_app.task(bind=True, acks_late=True)
def test_celery(self, timeout: int) -> str:
    from time import sleep

    for i in range(timeout):
        sleep(1)
        self.update_state(state="PROGRESS", meta={"current": i, "total": timeout})
    return f"test task return after timeout of {timeout} seconds"
