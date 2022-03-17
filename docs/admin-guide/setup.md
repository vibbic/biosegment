# Setup

Follow the instructions in the GitHub repo.

`README.md` gives a general overview of the software stack and installation requirements.

`EXTRA_INFORMATION.md` explains the software stack in more detail. It also contains notes on deploying the stack in a production environment.

Edit the `.env` file with the configuration needed for your installation.

A from scratch deployment on a `basic Ubuntu server 20.04 LTS` VM should have `Docker`, `pip`, `docker-compose` and `pip install docker-auto-labels`. Follow the documentation at [https://dockerswarm.rocks/](https://dockerswarm.rocks/).

Edit the following script to suit your use case:
```
bash scripts/start_swarm.sh
```

Build the application
```
TAG=prod FRONTEND_ENV=production bash ./scripts/build.sh
```

Deploy the application
```
DOMAIN=biosegment.ugent.be TRAEFIK_TAG=biosegment.ugent.be STACK_NAME=biosegment-ugent-be TAG=prod bash ./scripts/deploy.sh
```

Note that there are [two Traefik instances](https://github.com/tiangolo/full-stack-fastapi-postgresql/issues/240).
For more information, see
- [Initial server configuration guides](https://www.digitalocean.com/community/tutorials/initial-server-setup-with-ubuntu-20-04)
- [Docker Swarm guide](https://dockerswarm.rocks/)