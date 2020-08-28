from fastapi.testclient import TestClient
from sqlalchemy.orm import Session

from app.core.config import settings
from app.tests.utils.annotation import create_random_annotation


def test_create_annotation(
    client: TestClient, superuser_token_headers: dict, db: Session
) -> None:
    data = {"title": "Foo", "description": "Fighters"}
    response = client.post(
        f"{settings.API_V1_STR}/annotations/", headers=superuser_token_headers, json=data,
    )
    assert response.status_code == 200
    content = response.json()
    assert content["title"] == data["title"]
    assert content["description"] == data["description"]
    assert "id" in content
    assert "owner_id" in content


def test_read_annotation(
    client: TestClient, superuser_token_headers: dict, db: Session
) -> None:
    annotation = create_random_annotation(db)
    response = client.get(
        f"{settings.API_V1_STR}/annotations/{annotation.id}", headers=superuser_token_headers,
    )
    assert response.status_code == 200
    content = response.json()
    assert content["title"] == annotation.title
    assert content["description"] == annotation.description
    assert content["id"] == annotation.id
    assert content["owner_id"] == annotation.owner_id
