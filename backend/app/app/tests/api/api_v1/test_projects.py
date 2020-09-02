from fastapi.testclient import TestClient
from sqlalchemy.orm import Session

from app.core.config import settings
from app.tests.utils.project import create_random_project


def test_create_project(
    client: TestClient, superuser_token_headers: dict, db: Session
) -> None:
    data = {"title": "Foo", "description": "Fighters"}
    response = client.post(
        f"{settings.API_V1_STR}/projects/", headers=superuser_token_headers, json=data,
    )
    assert response.status_code == 200
    content = response.json()
    assert content["title"] == data["title"]
    assert content["description"] == data["description"]
    assert "id" in content
    assert "owner_id" in content


def test_read_project(
    client: TestClient, superuser_token_headers: dict, db: Session
) -> None:
    project = create_random_project(db)
    response = client.get(
        f"{settings.API_V1_STR}/projects/{project.id}", headers=superuser_token_headers,
    )
    assert response.status_code == 200
    content = response.json()
    assert content["title"] == project.title
    assert content["description"] == project.description
    assert content["id"] == project.id
    assert content["owner_id"] == project.owner_id
