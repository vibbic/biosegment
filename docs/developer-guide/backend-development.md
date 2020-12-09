# Backend development

The backend is implemented using [FastAPI](https://fastapi.tiangolo.com/). Read their documentation and also those of Pydantic and SQLAlchemy.

- backend hot-reloads on file changes
- Database schema changes need the removal of the database volume
- Backend tests

```bash
docker-compose exec backend bash /app/tests-start.sh
```

- Backend linting and formatting
    - via docker
    ```bash
    docker-compose exec backend /app/format-imports.sh
    docker-compose exec backend /app/lint.sh
    ```
    - locally
    ```bash
    cd backend/app
    poetry install
    poetry shell
    sh scripts/format-imports.sh
    sh scripts/lint.sh
    ```

## Structure

- `backend/app/app/`
    - `schemas/`
        - Pydantic schemas that define BioSegment data
    - `db/`
        - contains configuration for the database and initial setup
    - `models/`
        - SQLAlchemy models that define database table
        - uses the schemas
    - `crud/`
        - Python functions that implement database actions
        - uses the models
    - `api/`
        - endpoints that implement the API
        - uses the schemas to define input/output type
        - uses the crud actions
