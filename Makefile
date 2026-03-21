USER_NAME := $(shell whoami)
IMAGE_NAME := gaussiancar
TAG_NAME := v1
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

check-env:
ifndef PATH_TO_NUSCENES
	$(error PATH_TO_NUSCENES is undefined. Please run 'export PATH_TO_NUSCENES=/your/path' first)
endif
	@if [ ! -d "$(PATH_TO_NUSCENES)" ]; then \
		echo "Error: PATH_TO_NUSCENES directory does not exist at $(PATH_TO_NUSCENES)"; \
		exit 1; \
	fi

.PHONY: build run attach clear
build:
	docker build deploy/docker -t $(IMAGE_NAME):$(TAG_NAME) --build-arg USER=$(USER_NAME) --build-arg UID=$(UID) --build-arg GID=$(GID)
	@echo "\nBuild complete!"
	@echo "Run 'make run' to start the container."

run: check-env
	$(call run_docker, "source deploy/docker/entrypoint.sh && bash")

attach:
	docker exec -it $(CONTAINER_NAME) /bin/bash -c bash

clear:
	@rm -rf .cache/
	@rm -rf .venv/
	@rm -rf gaussiancar.egg-info/
	@find . -type d -name "__pycache__" -exec rm -rf {} +
	@rm -rf gaussiancar/ops/diff-gaussian-rasterization/build/
	@rm -rf gaussiancar/ops/diff-gaussian-rasterization.egg-info/
	@if [ -f "uv.lock" ]; then rm uv.lock; fi
	@echo "Cleaned up the project directory."
