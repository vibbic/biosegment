# Code generation

- The backend API can be used to generate code for interfaces and Axios calls with `openapi-generator`
  - download the latest API from `http://localhost/api/v1/openapi.json` and put it at `frontend/openapi.json`
  - Run the following line to update the code at `/frontend/api/generator/`.

```bash
# in project directory
docker run --rm -v $PWD:/local openapitools/openapi-generator-cli generate -i /local/openapi.json -g typescript-axios -o /local/frontend/src/api/generator/
```