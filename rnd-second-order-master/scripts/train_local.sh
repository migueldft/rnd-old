#!/bin/sh

DOCKERFILE_PATH=$1
DOCKER_IMAGE_NAME=$2

make build-image \
    DOCKERFILE_PATH=${DOCKERFILE_PATH}
    
docker run \
    -v $(pwd)/ml:/opt/ml --rm \
    ${DOCKER_IMAGE_NAME} train \
        --hyperparameters_file_name=hyperparameters.yaml
