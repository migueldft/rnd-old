#!/bin/sh

DOCKERFILE_PATH=$1
DOCKER_IMAGE_NAME=$2

make build-image \
    DOCKERFILE_PATH=${DOCKERFILE_PATH} \
    DOCKER_IMAGE_NAME=${DOCKER_IMAGE_NAME}
    
docker run \
    -it -v $(pwd)/ml:/opt/ml \
    -p 8080:8080 --rm \
    ${DOCKER_IMAGE_NAME} serve \
        --num_cpus=1