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
- For more information on running a GPU worker to create a segmentation, see the documentation.

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

See [Developer Guide]().

## Data folder
In `.env` a ROOT_DATA_FOLDER is defined with the default value of `./data`, relative to this project folder. The structure of the folder is documented in the [User Guide]().

Overwriting ROOT_DATA_FOLDER can be done using an environment variable:
```bash
# On Linux:
ROOT_DATA_FOLDER=/personal/data/folder/location docker-compose up -d --build

# On Windows:
set ROOT_DATA_FOLDER=X:/biosegment/data
docker-compose up -d --build
```

## Model training visualisation

We can use [TensorBoard](https://www.tensorflow.org/tensorboard) to visualize model (re)training.

Install Tensorboard with

```pip install tensorboard```

Then start tensorboard in the terminal with `logdir` pointing to the directory where biosegment stores the model training files, for example

```tensorboard --logdir "/home/johndoe/code/biosegment/data/models/EMBL/my retrained model 1"```

The tensorboard can be viewed in the browser at <http://localhost:6006>.

## Production

The `.env` file is version controlled, so it should not hold production secrets. Create from it a seperate `.env.prod` file with different secrets and credentials. This new file can be used instead of the default with an [`env-file` option parameter](https://docs.docker.com/compose/environment-variables/):
```bash
docker-compose --env-file .env.prod up
```

## References

- [Cookiecutter template: full-stack-fastapi-postgresql](https://github.com/tiangolo/full-stack-fastapi-postgresql/blob/master/README.md)
- [Dash: interactive image segmentation](https://github.com/plotly/dash-sample-apps/tree/master/apps/dash-image-segmentation)
- [Dash: 3D image partitioning](https://github.com/plotly/dash-3d-image-partitioning)