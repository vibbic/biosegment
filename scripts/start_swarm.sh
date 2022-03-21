#!/bin/bash

set -ex

# See https://dockerswarm.rocks/traefik/
docker swarm leave --force || true
docker swarm init
docker network rm traefik-public || true
docker network create --driver=overlay traefik-public
export NODE_ID=$(docker info -f '{{.Swarm.NodeID}}')
docker node update --label-add traefik-public.traefik-public-certificates=true $NODE_ID
export EMAIL=benjamin.rombaut@ugent.be
export DOMAIN=swarm.biosegment.ugent.be
export USERNAME=admin
export HASHED_PASSWORD=$(openssl passwd -apr1)
echo $HASHED_PASSWORD
# curl -L dockerswarm.rocks/traefik.yml -o traefik.yml
docker stack deploy -c traefik.yml traefik
docker stack ps traefik
