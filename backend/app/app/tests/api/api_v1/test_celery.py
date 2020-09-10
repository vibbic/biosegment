from typing import Dict

from fastapi.testclient import TestClient

from app.core.config import settings


def test_celery_worker_test(
    client: TestClient, superuser_token_headers: Dict[str, str]
) -> None:
    r = client.post(
        f"{settings.API_V1_STR}/utils/test-celery/?timeout=0",
        headers=superuser_token_headers,
    )
    assert r.status_code == 201
    response = r.json()
    assert response["task_id"]
