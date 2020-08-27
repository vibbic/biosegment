from typing import List

from fastapi import Depends, FastAPI, HTTPException
from sqlalchemy.orm import Session

from . import crud, models, schemas
from .db.session import SessionLocal, engine
from .db.base import Base

Base.metadata.create_all(bind=engine)

app = FastAPI(
    title="BioSegment backend",
    description="BioSegment backend handles datasets and their conversion.",
    version="0.1.0",
)

# Dependency
def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

@app.post("/users/{user_id}/projects/", response_model=schemas.Project)
def create_project(user_id: int, project: schemas.ProjectCreate, db: Session = Depends(get_db)):
    # db_project = crud.project.create_with_owner(db, owner_id=project.name)
    # if db_project:
    #     raise HTTPException(status_code=400, detail="Name already registered")
    return crud.project.create_with_owner(db=db, obj_in=project, owner_id=user_id)


@app.get("/users/{user_id}/projects/", response_model=List[schemas.Project])
def read_user_projects(owner_id: int, skip: int = 0, limit: int = 100, db: Session = Depends(get_db)):
    projects = crud.project.get_multi_by_owner(owner_id=owner_id, db=db, skip=skip, limit=limit)
    return projects

@app.get("/projects/{project_id}", response_model=schemas.Project)
def read_project(project_id: int, db: Session = Depends(get_db)):
    db_project = crud.project.get(db, id=project_id)
    if db_project is None:
        raise HTTPException(status_code=404, detail="Project not found")
    return db_project

@app.post("/users/", response_model=schemas.User)
def create_user(user: schemas.UserCreate, db: Session = Depends(get_db)):
    db_user = crud.user.get_by_email(db, email=user.email)
    if db_user:
        raise HTTPException(status_code=400, detail="Email already registered")
    return crud.user.create(db=db, obj_in=user)


@app.get("/users/", response_model=List[schemas.User])
def read_users(skip: int = 0, limit: int = 100, db: Session = Depends(get_db)):
    users = crud.user.get_multi(db)
    return users


@app.get("/users/{user_id}", response_model=schemas.User)
def read_user(user_id: int, db: Session = Depends(get_db)):
    db_user = crud.user.get(db, id=user_id)
    if db_user is None:
        raise HTTPException(status_code=404, detail="User not found")
    return db_user
