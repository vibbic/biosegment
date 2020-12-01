- backend hot-reloads on file changes
- Database schema changes need the removal of the database volume
- Backend tests

```bash
docker-compose exec backend bash /app/tests-start.sh
```

- Backend linting and formatting

```bash
cd backend/app
poetry install
poetry shell
sh scripts/format.sh
sh scripts/format-imports.sh
sh scripts/lint.sh
```
