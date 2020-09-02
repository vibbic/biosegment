from fastapi.testclient import TestClient
from sqlalchemy.orm import Session

from app.core.config import settings
from app.tests.utils.dataset import create_random_dataset


def test_create_dataset(
    client: TestClient, superuser_token_headers: dict, db: Session
) -> None:
    data = {"title": "Foo", "description": "Fighters"}
    response = client.post(
        f"{settings.API_V1_STR}/datasets/", headers=superuser_token_headers, json=data,
    )
    assert response.status_code == 200
    content = response.json()
    assert content["title"] == data["title"]
    assert content["description"] == data["description"]
    assert "id" in content
    assert "owner_id" in content


def test_read_dataset(
    client: TestClient, superuser_token_headers: dict, db: Session
) -> None:
    dataset = create_random_dataset(db)
    response = client.get(
        f"{settings.API_V1_STR}/datasets/{dataset.id}", headers=superuser_token_headers,
    )
    assert response.status_code == 200
    content = response.json()
    assert content["title"] == dataset.title
    assert content["description"] == dataset.description
    assert content["id"] == dataset.id
    assert content["owner_id"] == dataset.owner_id
