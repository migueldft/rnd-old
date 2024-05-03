ifdef AWS_REGION
AWS_REGION_PARAM = --region ${AWS_REGION}
endif

ifdef AWS_PROFILE
AWS_PROFILE_PARAM = --profile ${AWS_PROFILE}
endif

define get_version
$(shell git rev-parse --verify HEAD | sed -r 's/(.{8}).*/\1/g')
endef

CURRENT_UID := $(shell id -u)
CURRENT_GID := $(shell id -g)

TAG = $(call get_version,)
DOCKER_SERVER ?= "556684128444.dkr.ecr.us-east-1.amazonaws.com"
NAME = customer_segmentation
REPOSITORY = ${DOCKER_SERVER}/${NAME}
status=qa
local-company=gfg
STORAGE=s3://${local-company}-${status}-us-east-1-rnd-${NAME}


add-remote:
	dvc remove add -d remote ${STORAGE}

clean:  ## Clean python bytecodes
	@find . -name "*.pyc" | xargs rm -rf
	@find . -name "*.pyo" | xargs rm -rf
	@find . -name "__pycache__" -type d | xargs rm -rf
	docker system prune -a -f

test:
	scripts/test.sh $(opts) -- -x -q

run-rebase:
	git rebase -p master 2>/dev/null | grep "Your branch is up-to-date with 'origin/master'." || echo "\nPlease rebase your branch with master!"

build-image:
	docker build -t ${NAME}:${TAG} -t ${NAME}:latest -f Dockerfile .

# Initial commands for repo

get-data: build-image
	docker run \
		-u ${CURRENT_UID}:${CURRENT_UID} \
		-v ${PWD}/test_dir:/opt/ml \
		--rm ${NAME}:latest \
		python services/get_data.py 

dvc-get-data:
	dvc run \
		-o test_dir/input/data/training \
		-o test_dir/input/data/urls.gz \
		\
		-d Makefile -d Dockerfile -d src/services/get_data.py \
		\
		-d src/services/utils -d src/services/queries \
		-d test_dir/input/config/secrets.json \
		\
		-f get_data.dvc \
		make get-data

get-images: build-image
	docker run \
		-u ${CURRENT_UID}:${CURRENT_UID} \
		-v ${PWD}/test_dir:/opt/ml \
		--rm ${NAME}:latest \
		python services/get_images.py

dvc-get-images: build-image
	dvc run \
		-o test_dir/input/data/images \
		\
		-d Makefile -d Dockerfile \
		\
		-d src/services/utils \
		-d test_dir/input/data/urls.gz \
		\
		-d src/services/get_images.py \
		-f get_images.dvc \
		make get-images

train-local: build-image
	docker run -v ${PWD}/test_dir:/opt/ml --rm ${NAME}:latest task

dvc-train-local:
	dvc run \
		-d test_dir/input/data/training/data.csv \
		-d test_dir/input/config/hyperparameters.json \
		-d src/task \
		-o test_dir/model/model.joblib \
		-M test_dir/output/metrics.json \
		-f Dvcfile \
		make train-local

# Admin commands

docker-server-error:
ifndef DOCKER_SERVER
$(error DOCKER_SERVER variable must be set)
endif

docker-login: docker-server-error
	$$(eval "$$(echo aws ecr get-login --no-include-email ${AWS_REGION_PARAM} ${AWS_PROFILE_PARAM})")

release-image: docker-login
	docker tag ${NAME}:${TAG} ${REPOSITORY}:${TAG} && \
		docker push $(REPOSITORY):${TAG}

pull-image: docker-login
	docker pull $(REPOSITORY):${TAG}

release-image-latest: docker-login
	docker tag ${REPOSITORY}:${TAG} ${REPOSITORY}:latest && \
		docker push $(REPOSITORY):latest

helm-deploy:
	@aws lambda invoke --function-name helm-deployer --payload "{ \"name\": \"${NAME}\", \"tillerNamespace\": \"${NAME}-helm\", \"tag\": \"${TAG}\", \"timeout\": 1200, \"chartFolder\": \"deploy/chart\", \"base64Kubeconfig\": \"$(KUBECONFIG)\", \"privateChart\": true, \"privateChartType\": \"gfg-ml-batch\", \"privateChartVersion\": \"0.2.1\" }" /dev/stdout

tag-version:
	git tag $(TAG) && git push --force --tags

promote-image: docker-login
	docker tag ${REPOSITORY}:${TAG} ${TO}:${TAG}
	docker push ${TO}:${TAG}