from fastapi.testclient import TestClient
from sqlalchemy.orm import Session

from app.core.config import settings
from app.tests.utils.dataset import create_random_dataset
from app.tests.utils.project import create_random_project


def test_create_project_dataset(
    client: TestClient, superuser_token_headers: dict, db: Session
) -> None:
    project = create_random_project(db)
    project_id = project.id
    data = {"title": "Foo", "description": "Fighters"}
    response = client.post(
        f"{settings.API_V1_STR}/projects/{project_id}/datasets",
        headers=superuser_token_headers,
        json=data,
    )
    assert response.status_code == 200
    content = response.json()
    assert content["title"] == data["title"]
    assert content["description"] == data["description"]
    assert "id" in content
    assert "owner_id" in content


def test_read_project_dataset(
    client: TestClient, superuser_token_headers: dict, db: Session
) -> None:
    dataset = create_random_dataset(db)
    project_id = dataset.project.id
    dataset_id = dataset.id
    response = client.get(
        f"{settings.API_V1_STR}/datasets/{dataset_id}", headers=superuser_token_headers,
    )
    assert response.status_code == 200
    content = response.json()
    assert content["title"] == dataset.title
    assert content["description"] == dataset.description
    assert content["id"] == dataset.id
    assert content["owner_id"] == dataset.owner_id
