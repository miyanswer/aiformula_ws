.PHONY: all build up down stop restart bash root-bash logs clean gui open-gui \
        yolo-install record extract split train detect

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
# YOLO Pipeline Commands
# ==============================================================================

# Install Python requirements for YOLO and video tools
yolo-install:
	pip3 install -r requirements-yolo.txt

# Record 15fps FHD video from webcam
record:
	python3 scripts/record_video.py

# Extract training frames from recorded videos
extract:
	python3 scripts/extract_frames.py

# Split annotated images/labels and create data.yaml
split:
	python3 scripts/split_dataset.py

# Train YOLO model (default: yolo11n.pt, 50 epochs)
train:
	python3 scripts/train_yolo.py

# Real-time YOLO detection using webcam
detect:
	python3 scripts/detect_webcam.py

