from fastapi.testclient import TestClient
from sqlalchemy.orm import Session

from app.core.config import settings
from app.tests.utils.model import create_random_model


def test_create_model(
    client: TestClient, superuser_token_headers: dict, db: Session
) -> None:
    data = {"title": "Foo", "description": "Fighters"}
    response = client.post(
        f"{settings.API_V1_STR}/models/", headers=superuser_token_headers, json=data,
    )
    assert response.status_code == 200
    content = response.json()
    assert content["title"] == data["title"]
    assert content["description"] == data["description"]
    assert "id" in content
    assert "owner_id" in content


def test_read_model(
    client: TestClient, superuser_token_headers: dict, db: Session
) -> None:
    model = create_random_model(db)
    response = client.get(
        f"{settings.API_V1_STR}/models/{model.id}", headers=superuser_token_headers,
    )
    assert response.status_code == 200
    content = response.json()
    assert content["title"] == model.title
    assert content["description"] == model.description
    assert content["id"] == model.id
    assert content["owner_id"] == model.owner_id
