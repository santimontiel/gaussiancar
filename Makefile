USER_NAME := $(shell whoami)
IMAGE_NAME := gaussiancar
TAG_NAME := v1.0.0
CONTAINER_NAME := $(IMAGE_NAME)_container
GPU_ID := 0

UID := $(shell id -u)
GID := $(shell id -g)

WANDB_API_KEY := $(shell echo $$WANDB_API_KEY)
PATH_TO_NUSCENES := $(shell echo $$PATH_TO_NUSCENES)

define run_docker
	@docker run -it --rm \
		--net host \
		--gpus '"device=$(GPU_ID)"' \
		--ipc host \
		--ulimit memlock=-1 \
		--ulimit stack=67108864 \
		--name=$(CONTAINER_NAME) \
		-u $(USER_NAME) \
		-v ./:/workspace \
		-v $(PATH_TO_NUSCENES):/data/nuscenes \
		-e WANDB_API_KEY=$(WANDB_API_KEY) \
		-e TERM=xterm-256color \
		$(IMAGE_NAME):$(TAG_NAME) \
		/bin/bash -c $(1)
endef

.PHONY: build run attach clear
build:
	docker build . -t $(IMAGE_NAME):$(TAG_NAME) --build-arg USER=$(USER_NAME) --build-arg UID=$(UID) --build-arg GID=$(GID)
	@echo "\nBuild complete!"
	@echo "Run 'make run' to start the container."

run:
	$(call run_docker, "source entrypoint.sh && bash")

attach:
	docker exec -it $(CONTAINER_NAME) /bin/bash -c bash

clear:
	@rm -rf .venv/
	@rm -rf gaussiancar.egg-info/
	@find . -type d -name "__pycache__" -exec rm -rf {} +
	@rm -rf gaussiancar/ops/diff-gaussian-rasterization/build/
	@rm -rf gaussiancar/ops/diff-gaussian-rasterization.egg-info/
	@echo "Cleaned up the project directory."