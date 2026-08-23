.PHONY: all build up down stop restart bash root-bash logs clean gui open-gui \
        yolo-install record extract label auto-annotate auto-annotate-sam split train detect

# Default target
all: up

# Build or rebuild Docker image
build:
	docker compose build

# Start container in background
up:
	docker compose up -d

# Stop container
stop:
	docker compose stop

# Stop and remove containers, networks
down:
	docker compose down

# Restart container
restart: down up

# Open a bash shell inside the container as rosuser
bash:
	docker compose exec aiformula_ws bash

# Open a bash shell inside the container as root
root-bash:
	docker compose exec -u root aiformula_ws bash

# Open Web GUI (noVNC) in default browser
gui open-gui:
	open http://localhost:8080

# View container logs
logs:
	docker compose logs -f aiformula_ws

# Clean build artifacts
clean:
	rm -rf build install log

# ==============================================================================
# YOLO Multi-Task Pipeline Commands (e.g. make train TASK=traffic_light)
# ==============================================================================

TASK ?=

# Install Python requirements for YOLO and video tools
yolo-install:
	pip3 install -r requirements-yolo.txt

# Record 15fps FHD video from webcam (e.g. make record TASK=traffic_light)
record:
	python3 scripts/record_video.py $(if $(TASK),--task $(TASK),)

# Extract training frames from recorded videos (e.g. make extract TASK=traffic_light)
extract:
	python3 scripts/extract_frames.py $(if $(TASK),--task $(TASK),)

# One-Shot Auto-Annotation (Label 1st image, auto-annotate all other frames for free)
auto-annotate:
	python3 scripts/auto_annotate.py $(if $(TASK),--task $(TASK),)

# One-Shot Auto-Annotation with SAM 2 AI refinement
auto-annotate-sam:
	python3 scripts/auto_annotate.py --use-sam $(if $(TASK),--task $(TASK),)

# Launch AnyLabeling annotation tool for manual labeling
label:
	@if [ -n "$(TASK)" ]; then \
		mkdir -p data/tasks/$(TASK)/extracted_frames; \
		python3 -m anylabeling.app data/tasks/$(TASK)/extracted_frames; \
	else \
		python3 -m anylabeling.app data/tasks/jinmen_dog/extracted_frames; \
	fi

# Split annotated images/labels and create data.yaml (e.g. make split TASK=traffic_light)
split:
	python3 scripts/split_dataset.py $(if $(TASK),--task $(TASK),)

# Train YOLO model (e.g. make train TASK=traffic_light)
train:
	python3 scripts/train_yolo.py $(if $(TASK),--task $(TASK),)

# Real-time YOLO detection using webcam (e.g. make detect TASK=traffic_light)
detect:
	python3 scripts/detect_webcam.py $(if $(TASK),--task $(TASK),)

