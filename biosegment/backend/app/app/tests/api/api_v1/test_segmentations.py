from fastapi.testclient import TestClient
from sqlalchemy.orm import Session

from app.core.config import settings
from app.tests.utils.segmentation import create_random_segmentation


def test_create_segmentation(
    client: TestClient, superuser_token_headers: dict, db: Session
) -> None:
    data = {"title": "Foo", "description": "Fighters"}
    response = client.post(
        f"{settings.API_V1_STR}/segmentations/",
        headers=superuser_token_headers,
        json=data,
    )
    assert response.status_code == 200
    content = response.json()
    assert content["title"] == data["title"]
    assert content["description"] == data["description"]
    assert "id" in content
    assert "owner_id" in content


def test_read_segmentation(
    client: TestClient, superuser_token_headers: dict, db: Session
) -> None:
    segmentation = create_random_segmentation(db)
    response = client.get(
        f"{settings.API_V1_STR}/segmentations/{segmentation.id}",
        headers=superuser_token_headers,
    )
    assert response.status_code == 200
    content = response.json()
    assert content["title"] == segmentation.title
    assert content["description"] == segmentation.description
    assert content["id"] == segmentation.id
    assert content["owner_id"] == segmentation.owner_id
