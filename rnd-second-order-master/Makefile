DOCKER_IMAGE_NAME=second-order


.PHONY: help clean clean-pyc clean-build list test test-dbg test-cov coverage-report test-all docs release sdist install deps develop tag

define get_git_version
$(shell git rev-parse --verify HEAD | sed -r 's/(.{7}).*/\1/g')
endef

define get_rollback_version
$(shell (git describe --tags `git rev-list --tags --max-count=2` | tail -n1))
endef

TIME_STAMP=$(shell date "+%Y-%m-%d-%H-%M-%S")

IMAGE_TAG ?= $(call get_git_version,)
IMAGE_TAG_PUSH ?= ${IMAGE_TAG}

TRAINING_JOB_NAME := "${DOCKER_IMAGE_NAME}-${IMAGE_TAG_PUSH}-${TIME_STAMP}"

inputdataconfig_file=ml/input/config/inputdataconfig.json
INPUTDATACONFIG=`cat ${inputdataconfig_file}`
hyperparameters_file=ml/input/config/hyperparameters.json
HYPERPARAMETERS=`cat ${hyperparameters_file}`
resourceconfig_file=ml/input/config/resourceconfig.json
RESOURCECONFIG=`cat ${resourceconfig_file}`
stoppingcondition_file=ml/input/config/stoppingcondition.json
STOPPINGCONDITION=`cat ${stoppingcondition_file}`
query_file=data_etl/query.sql

help:
	@echo "clean-build - remove build artifacts"
	@echo "clean-pyc - remove Python file artifacts"
	@echo "lint - check style with flake8"
	@echo "test - run tests quickly with the default Python"
	@echo "test-cov - run tests with the default Python and report coverage"
	@echo "coverage-report - run tests with the default Python and create html report"
	@echo "test-dbg - run tests and debug with pdb"
	@echo "testloop - run tests in a loop"
	@echo "docs - generate Sphinx HTML documentation, including API docs"
	@echo "release - package and upload a release"
	@echo "sdist - package"
	@echo "install - install"
	@echo "develop - install in development mode"
	@echo "deps - install dependencies"
	@echo "dev_deps - install dependencies for development"
	@echo "release - package a release in wheel and tarball"
	@echo "upload - make a release and run the scripts/deploy.sh"
	@echo "tag - create a git tag with current version"

install:
	pipenv run python setup.py install

develop: dev_deps
	pipenv run python setup.py develop

deps:
	pipenv install

dev_deps:
	pipenv install --dev

clean: clean-build clean-pyc clean-docker

clean-build:
	rm -fr build/
	rm -fr dist/
	rm -fr *.egg-info

clean-pyc:
	find . -name '*.pyc' -exec rm -f {} +
	find . -name '*.pyo' -exec rm -f {} +
	find . -name '*~' -exec rm -f {} +
	find . -name '__pycache__' -exec rm -rf {} +
	find . -name '*.log*' -delete

clean-docker:
	docker image prune --force

python-lint:
	pipenv run flake8 src/

docker-lint: check-env-docker


lint: python-lint

test:
	pipenv run py.test -v

test-cov:
	pipenv run pytest --cov-report=term-missing --cov=src

coverage-report:
	pipenv run pytest --cov-report=term-missing --cov=src
	pipenv run coverage html

test-dbg:
	pipenv run py.test --pdb

testloop:
	pipenv run py.test -f

build:
	pipenv run python setup.py sdist --formats gztar bdist_wheel

tag: clean
	@echo "Creating git tag v$(VERSION)"
	git tag v$(VERSION)
	git push --tags

release: clean build tag
	pipenv run python setup.py sdist upload

patch:
	pipenv run bumpversion patch

minor:
	pipenv run bumpversion minor

major:
	pipenv run bumpversion major


## Docker commands

check-env-docker:
ifndef DOCKERFILE_PATH
	$(error DOCKERFILE_PATH is undefined)
endif

build-image: check-env-docker
	docker build -f ${DOCKERFILE_PATH} -t ${DOCKER_IMAGE_NAME} .


## local tests

check-local-qry:
ifeq ("$(wildcard $(query_file))", "")
	$(error ${query_file} is non-existent)
endif

data: check-local-qry
	./data_etl/get_data.sh

train:
	./scripts/train_local.sh docker/Dockerfile.training ${DOCKER_IMAGE_NAME}

train-optimize:
	./scripts/train_local_optimize.sh docker/Dockerfile.training ${DOCKER_IMAGE_NAME}

serve:
	./scripts/serve_local.sh docker/Dockerfile.training ${DOCKER_IMAGE_NAME}

predict:
	./scripts/predict_local.sh ml/input/api/payload.csv text/csv > test_csv.log
	./scripts/predict_local.sh ml/input/api/payload.json application/json > test_json.log


## Deployment commands

check-env-aws:
ifndef AWS_ACCESS_KEY_ID
	$(error AWS_ACCESS_KEY_ID is undefined)
endif
ifndef AWS_SECRET_ACCESS_KEY
	$(error AWS_SECRET_ACCESS_KEY is undefined)
endif
ifndef AWS_DEFAULT_REGION
	$(error AWS_DEFAULT_REGION is undefined)
endif

check-server-docker:
ifndef DOCKER_SERVER
	$(error DOCKER_SERVER is undefined)
endif

docker-login: check-env-aws check-server-docker
	aws ecr get-login-password --region ${AWS_DEFAULT_REGION} | docker login --username AWS --password-stdin ${DOCKER_SERVER}/${DOCKER_IMAGE_NAME}

release-image: docker-login
	docker tag ${DOCKER_IMAGE_NAME} ${DOCKER_SERVER}/${DOCKER_IMAGE_NAME}:${IMAGE_TAG_PUSH} && \
		docker push ${DOCKER_SERVER}/${DOCKER_IMAGE_NAME}:${IMAGE_TAG_PUSH}

check-env-sagemaker:
ifndef AWS_PROJECT_BUCKET
	$(error AWS_PROJECT_BUCKET is undefined)
endif
ifndef AWS_SAGEMAKER_ROLE
	$(error AWS_SAGEMAKER_ROLE is undefined)
endif

check-parameter-files:
ifeq ("$(wildcard $(inputdataconfig_file))", "")
	$(error ${inputdataconfig_file} is non-existent)
endif
ifeq ("$(wildcard $(resourceconfig_file))", "")
	touch ${resourceconfig_file}
	echo "{\"InstanceType\"=\"ml.m4.xlarge\",\"InstanceCount\"=1,\"VolumeSizeInGB\"=10}" > ${resourceconfig_file}
endif
ifeq ("$(wildcard $(hyperparameters_file))", "")
	touch ${hyperparameters_file}
	echo "{}" > ${hyperparameters_file}
endif
ifeq ("$(wildcard $(stoppingcondition_file))", "")
	touch ${stoppingcondition_file}
	echo "{\"MaxRuntimeInSeconds\": 3600}" > ${stoppingcondition_file}
endif

sagemaker-training-job: check-env-aws check-env-sagemaker check-parameter-files
	sed -i 's/{AWS_PROJECT_BUCKET}/${AWS_PROJECT_BUCKET}/g' ${inputdataconfig_file}
	aws sagemaker create-training-job \
		--training-job-name="${TRAINING_JOB_NAME}" \
		--algorithm-specification="TrainingImage=${DOCKER_SERVER}/${DOCKER_IMAGE_NAME}:${IMAGE_TAG_PUSH},TrainingInputMode=File" \
		--role-arn="${AWS_SAGEMAKER_ROLE}" \
		--output-data-config="S3OutputPath=s3://${AWS_PROJECT_BUCKET}/training-jobs/" \
		--resource-config="${RESOURCECONFIG}" \
		--stopping-condition="${STOPPINGCONDITION}" \
		--input-data-config="${INPUTDATACONFIG}" \
		--hyper-parameters="${HYPERPARAMETERS}"
		--tags Key=system,Value="${DOCKER_IMAGE_NAME}" Key=role,Value=machine_learning Key=group,Value=rnd Key=env,Value=experimentation Key=company,Value=gfg Key=type,Value=service
