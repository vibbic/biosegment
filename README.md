# BioSegment

BioSegment is a software stack to enable segmentation of microscopy data using machine learning models.

## Supported platforms

- Ubuntu 18.04, 20.04
- Windows 10

## Installation

Install:
- [Docker](https://docs.docker.com/get-docker/)
- [Conda](https://docs.conda.io/en/latest/miniconda.html)
- [docker-compose](https://docs.docker.com/compose/install/)
  - at least `version 1.26.2, build eefe0d31`

Note that the Docker images take up ~8GB of disk space.
Create a data folder according to the specification or use the provided script to create one at `data/`. Edit `data/setup.json` to add more datasets.
```
bash scripts/create-data-folder.sh
```

Create and run the BioSegment stack using `docker-compose`. The first time could take ~10 minutes.
```bash
docker-compose up -d --build
```

- BioSegment is now running in dev mode at localhost
- Go to `http://localhost/dash/viewer` to see the Dash viewer on the default dataset.

## Overview

- `dash_frontend/` for dashboard and segmentation viewer, using Python Dash: <http://localhost/dash>
- `frontend/` for account managment, built with Javascript Vue, with routes handled based on the path: <http://localhost>
- `backend/`, JSON based web API using Python FastAPI based on OpenAPI: <http://localhost/api/>
  - Automatic interactive documentation with Swagger UI (from the OpenAPI backend): <http://localhost/docs>
  - Alternative automatic documentation with ReDoc (from the OpenAPI backend): <http://localhost/redoc>
- `gpu_worker/` Python Celery task runner to run PyTorch models (e.g. [neuralnets](https://pypi.org/project/neuralnets/)) directly on the host GPU
- `diagrams/` with system diagrams using [mermaid](https://mermaid-js.github.io/mermaid/)
- extra
  - PGAdmin, PostgreSQL web administration: <http://localhost:5050>
  - Flower, administration of Celery tasks: <http://localhost:5555>
  - Traefik UI, to see how the routes are being handled by the proxy: <http://localhost:8090>
- `.env` contains all login credentials and configurations

## Development

### Frontend development

- Install locally [Node](https://nodejs.org/en/download/) and npm
- Enter the `frontend` directory, install the NPM packages and start the live server using the `npm` scripts:

```bash
cd frontend
npm install
npm run serve
```

Then open your browser at http://localhost:8080

Notice that this live server is not running inside Docker, it is for local development, and that is the recommended workflow. Once you are happy with your frontend, you can build the frontend Docker image and start it, to test it in a production-like environment. But compiling the image at every change will not be as productive as running the local development server with live reload.

Check the file `package.json` to see other available options.

```bash
# unit test
npm run unit:test
# lint
npm run lint
```

If you have Vue CLI installed, you can also run `vue ui` to control, configure, serve, and analyze your application using a nice local web user interface.

### Dash frontend development

- Dash frontend hot-reloads on file changes
- The backend API can be used to generate code for interfaces and Axios calls with `openapi-generator`
  - download the latest API from `http://localhost/api/v1/openapi.json` and put it at `frontend/openapi.json`
  - Run the following line to update the code at `/frontend/api/generator/`.

```bash
# in project directory
docker run --rm -v $PWD:/local openapitools/openapi-generator-cli generate -i /local/openapi.json -g typescript-axios -o /local/frontend/src/api/generator/
```

### Backend development

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

## Data folder
In `.env` a ROOT_DATA_FOLDER is defined with the default value of `./data`, relative to this project folder. The structure of the folder is the following:
- ROOT_DATA_FOLDER
  - setup.json
    - JSON file containing configurations the backend reads during initialization of the database. Used in development.
  - EM/
    - {dataset_name} e.g. EMBL
      - raw/
        - {pngs}
  - models/
    - {model_name} e.g. unet_2d
      - = output folder of neuralnets training
      - saved model e.g. best_checkpoint.pytorch
  - segmentations/
    - {dataset_name}
      - labels/
        - = ground truth labels of the dataset
        - {pngs}
      - {segmentation_name}
        - = output folder of neuralnets inference
        - {pngs}
  - annotations
    - {dataset_name}
      - {annotation_name}
        - saved annotations e.g. annotations.json

In the future the backend will handle this folder structure.
Overwriting ROOT_DATA_FOLDER can be done using an environment variable:
```bash
# On Linux:
ROOT_DATA_FOLDER=/personal/data/folder/location docker-compose up -d --build

# On Windows:
set ROOT_DATA_FOLDER=X:/biosegment/data
docker-compose up -d --build
```

## Run celery worker with GPU

GPU support in docker-compose is very experimental, not working currently
- see `docker-compose_gpu.yml`
- docker-compose override gives errors, that's why one .yml file is needed
- NVIDIA driver still isn't visible then, waiting for stable support

Current workaround
- expose rabbitMQ queue in docker-compose to host
- run celery worker on host without virtualization

```bash
cd gpu_worker

# install environment for neuralnets celery worker
conda env update -f celery_all_environment.yaml
conda activate celery_neuralnets

# On Linux
bash start_worker.sh

# On Windows
set ROOT_DATA_FOLDER=X:/biosegment/data
start_worker.bat   # On Windows
```

If force stopping the auto-reloading watchdog for workers (x2 Ctrl-C), some workers may linger.
This will show up as warning when a new worker with the same name is started.

View all host celery workers
```bash
ps aux|grep 'celery worker'
```

Kill them all
```bash
ps auxww | grep 'celery worker' | awk '{print $2}' | xargs kill -9
```

## Model training visualisation

We can use [TensorBoard](https://www.tensorflow.org/tensorboard) to visualize model (re)training.

Install Tensorboard with

```pip install tensorboard```

Then start tensorboard in the terminal with `logdir` pointing to the directory where biosegment stores the model training files, for example

```tensorboard --logdir "/home/johndoe/code/biosegment/data/models/EMBL/my retrained model 1"```

The tensorboard can be viewed in the browser at <http://localhost:6006>.

## CI

Continuous integration using [GitHub Actions](https://docs.github.com/en/free-pro-team@latest/actions) can run linting or tests on code changes. The configuration files are located in `.github/`.

For local development, [act](https://github.com/nektos/act) can be used.

```bash
# install act
curl https://raw.githubusercontent.com/nektos/act/master/install.sh | sudo bash
```

Example commands:
```bash
act -P ubuntu-latest=berombau/act_base -j test
# WARNING act-environments-ubuntu:18.04 is >18GB!
act -P ubuntu-latest=nektos/act-environments-ubuntu:18.04 -j lint
```

Caveats
- There are differences between `berombau/act_base` and the GitHub images
- different oses are not supported
- public GitHub actions can depend on certain GitHub tooling, which would require incorporating that dependency in the `act_base` image.

## Production

The `.env` file is version controlled, so it should not hold production secrets. Create from it a seperate `.env.prod` file with different secrets and credentials. This new file can be used instead of the default with an [`env-file` option parameter](https://docs.docker.com/compose/environment-variables/):
```bash
docker-compose --env-file .env.prod up
```

## References

- [Cookiecutter template: full-stack-fastapi-postgresql](https://github.com/tiangolo/full-stack-fastapi-postgresql/blob/master/README.md)
- [Dash: interactive image segmentation](https://github.com/plotly/dash-sample-apps/tree/master/apps/dash-image-segmentation)
- [Dash: 3D image partitioning](https://github.com/plotly/dash-3d-image-partitioning)