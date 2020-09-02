# Cheatsheet

Overview of the most common commands.

## Installation

Install:
- Docker
- Conda

- Install `biosegment` conda environment
```
conda env remove -n biosegment
conda env update -f environment.yaml --prune
conda activate biosegment
```

- Use the docker-compose in the environment
```
docker-compose up -d --build
```

- BioSegment is now running in dev mode at localhost

## Overview
Dash-frontend, using Python Dash: http://localhost/dash

Frontend, built with Docker, with routes handled based on the path: http://localhost

Backend, JSON based web API based on OpenAPI: http://localhost/api/

Automatic interactive documentation with Swagger UI (from the OpenAPI backend): http://localhost/docs

Alternative automatic documentation with ReDoc (from the OpenAPI backend): http://localhost/redoc

PGAdmin, PostgreSQL web administration: http://localhost:5050

Flower, administration of Celery tasks: http://localhost:5555

Traefik UI, to see how the routes are being handled by the proxy: http://localhost:8090

## Development
- Dash frontend and backend hotreload on file changes
- Database schema changes need the removal of the database volume
- Backend tests
```
docker-compose exec backend bash /app/tests-start.sh
```
- Backend linting and formatting
```
cd backend/app
poetry install
poetry shell
sh scripts/format.sh
sh scripts/format-imports.sh
sh scripts/lint.sh
```