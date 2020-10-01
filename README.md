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
- Dash frontend and backend hot-reload on file changes
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

## Data folder
In `.env` a ROOT_DATA_FOLDER is defined with the default value of `./data`, relative to this project folder. The structure of the folder is the following:
- ROOT_DATA_FOLDER
    - EM
        - EMBL
            - raw
                - .pngs
            - labels
    - models
        - unet_2d (output folder of neuralnets training)
            - best_checkpoint.pytorch
    - segmentations
        - EMBL (output folder of neuralnets inference)
            - .pngs
    - annotations

In the future the backend will handle this folder structure.
Overwriting ROOT_DATA_FOLDER can be done using an environment variable:
```
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
```
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
```
ps aux|grep 'celery worker'
```

Kill them all
```
ps auxww | grep 'celery worker' | awk '{print $2}' | xargs kill -9
```

## Model training visualisation

We can use [TensorBoard](https://www.tensorflow.org/tensorboard) to visualize model (re)training.

Install Tensorboard with

```pip install tensorboard```

Then start tensorboard in the terminal with `logdir` pointing to the directory where biosegment stores the model training files, for example 

```tensorboard --logdir "/home/johndoe/code/biosegment/data/models/EMBL/my retrained model 1"``` 

The tensorboard can be viewed in the browser at http://localhost:6006.

## CI

Continuous integration using [GitHub Actions](https://docs.github.com/en/free-pro-team@latest/actions) can run linting or tests on code changes. The configuration files are located in `.github/`.

For local development, [act](https://github.com/nektos/act) can be used.

```bash
# install act
curl https://raw.githubusercontent.com/nektos/act/master/install.sh | sudo bash
```

Example commands:
```
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
```
docker-compose --env-file .env.prod up 
```
