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

# Build ROS 2 workspace packages inside Docker container
build-ws:
	docker compose exec aiformula_ws bash -c "source /opt/ros/humble/setup.bash && colcon build --symlink-install"

# [PC Standalone Test] Run full pipeline test with video input & RViz inside Docker
test-pc:
	docker compose exec aiformula_ws bash -c "source /opt/ros/humble/setup.bash && source install/setup.bash && ros2 launch sample_launchers pc_standalone_test.launch.py use_device:=cpu rviz:=true"

# [Real Vehicle Full Bringup] Run all hardware + autonomous driving nodes
bringup-all:
	docker compose exec aiformula_ws bash -c "source /opt/ros/humble/setup.bash && source install/setup.bash && bash src/aiformula_base/bash/3_bringup_all_nodes.sh"

# [Real Vehicle Hardware Bringup] Run hardware nodes only
bringup-hw:
	docker compose exec aiformula_ws bash -c "source /opt/ros/humble/setup.bash && source install/setup.bash && bash src/aiformula_base/bash/1_bringup_hardware.sh"

# Stop & kill all lingering ROS 2 and RViz nodes inside container
stop-nodes kill-nodes:
	docker compose exec aiformula_ws bash -c "pkill -9 -f 'ros2|rviz2|video_publisher|yolop_lane_detector|bev_pure_pursuit_node|robot_state_publisher|joint_state_publisher' || true"

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

# Extract training frames from videos and import static images (e.g. make extract TASK=traffic_light)
extract:
	python3 scripts/extract_frames.py $(if $(TASK),--task $(TASK),) $(if $(filter 1 true,$(BEV)),--bev,)

# Extract training frames with Bird's Eye View (BEV / IPM) transformation (e.g. make extract-bev TASK=t_junction)
extract-bev:
	python3 scripts/extract_frames.py $(if $(TASK),--task $(TASK),) --bev --every-sec 0.5 --bev-save-video

# Import static images from raw_images/ (e.g. make import-images TASK=traffic_light)
import-images:
	python3 scripts/extract_frames.py $(if $(TASK),--task $(TASK),) --video ""

# One-Shot Auto-Annotation (Label 1st image, auto-annotate all other frames for free)
auto-annotate:
	python3 scripts/auto_annotate.py $(if $(TASK),--task $(TASK),)

# One-Shot Auto-Annotation with SAM 2 AI refinement
auto-annotate-sam:
	python3 scripts/auto_annotate.py --use-sam $(if $(TASK),--task $(TASK),)

# Pseudo-labeling / Auto-labeling with trained YOLO model (Fast auto-labeling for remaining frames)
pseudo-label auto-label:
	python3 scripts/pseudo_label.py $(if $(TASK),--task $(TASK),)

# Launch AnyLabeling annotation tool for manual labeling
label:
	@if [ -n "$(TASK)" ]; then \
		mkdir -p data/tasks/$(TASK)/raw_images data/tasks/$(TASK)/raw_videos data/tasks/$(TASK)/extracted_frames; \
		python3 -m anylabeling.app data/tasks/$(TASK)/extracted_frames; \
	else \
		mkdir -p data/tasks/jinmen_dog/raw_images data/tasks/jinmen_dog/raw_videos data/tasks/jinmen_dog/extracted_frames; \
		python3 -m anylabeling.app data/tasks/jinmen_dog/extracted_frames; \
	fi

# Split annotated images/labels and create data.yaml (e.g. make split TASK=traffic_light)
split:
	python3 scripts/split_dataset.py $(if $(TASK),--task $(TASK),)

# Train YOLO model (e.g. make train TASK=traffic_light)
train:
	python3 scripts/train_yolo.py $(if $(TASK),--task $(TASK),)

VIDEO ?=
SAVE ?=

# Real-time YOLO detection using webcam (e.g. make detect TASK=traffic_light)
detect:
	python3 scripts/detect_webcam.py $(if $(TASK),--task $(TASK),)

# Test YOLO detection on video files with playback & controls (e.g. make detect-video TASK=traffic_light)
detect-video:
	python3 scripts/detect_video.py $(if $(TASK),--task $(TASK),) $(if $(VIDEO),--video $(VIDEO),) $(if $(SAVE),--save,)

# ==============================================================================
# YOLOP Lane Segmentation Pipeline Commands (Fine-Tuning)
# ==============================================================================

# Extract frames from mp4/ videos and auto-generate initial pseudo masks
prepare-yolop-data:
	python3 scripts/prepare_yolop_dataset.py $(if $(VIDEO),--video-file $(VIDEO),)

# Process existing masks (e.g. make mask-top or make crop-bottom)
mask-top:
	python3 scripts/process_masks.py --mode mask_top --top-cut-ratio 0.45

crop-bottom:
	python3 scripts/process_masks.py --mode crop_bottom --top-cut-ratio 0.45

clean-noise:
	python3 scripts/process_masks.py --mode clean_only --clean-noise --min-area 30

# Fine-tune YOLOP lane segmentation model using data/yolop_dataset
train-yolop:
	python3 scripts/train_lane_yolop.py

# Fine-tune with Method A (Top Masking) - Standard 30 epochs
train-yolop-mask:
	python3 scripts/train_lane_yolop.py --mask-top --top-cut-ratio 0.45 --output-name shiho_lane_mask

# 🔥 Fine-tune with Method A (Top Masking) - STABILIZED & EXTENDED (Warmup + Batch 8 + 50 epochs)
train-yolop-mask-v2:
	python3 scripts/train_lane_yolop.py --mask-top --top-cut-ratio 0.45 --epochs 50 --batch-size 8 --lr 1.5e-4 --warmup-epochs 3 --output-name shiho_lane_mask_v2

# Fine-tune with Method B (Bottom Cropping - 2x Resolution)
train-yolop-crop:
	python3 scripts/train_lane_yolop.py --crop-bottom --top-cut-ratio 0.45 --output-name shiho_lane_crop

# Fine-tune from Official Untouched YOLOP (BDD100K) with Bottom Cropping
train-official-crop:
	python3 scripts/train_lane_yolop.py --weights models/pretrained/yolop_official.pth --crop-bottom --top-cut-ratio 0.45 --output-name yolop_official_crop

# Fine-tune from Official Untouched YOLOP (BDD100K) with Top Masking
train-official-mask:
	python3 scripts/train_lane_yolop.py --weights models/pretrained/yolop_official.pth --mask-top --top-cut-ratio 0.45 --output-name yolop_official_mask

WEIGHTS ?=

# Interactive Lane Detection Debugger on Video (e.g. make debug-lane or make debug-lane VIDEO=mp4/xxx.mp4)
debug-lane eval-yolop:
	python3 scripts/eval_lane_yolop.py $(if $(WEIGHTS),--weights $(WEIGHTS),) $(if $(VIDEO),--video $(VIDEO),) $(if $(SAVE),--save-video,)

# Run 2027 BEV Controller Video Test inside Docker
test-control test-bev:
	docker compose exec aiformula_ws bash -c "source /opt/ros/humble/setup.bash && source install/setup.bash && ros2 launch ai_formula_oit_2027 video_bev_control_test.launch.py use_device:=cpu"


# Run Traffic Light Detection & Depth Estimation on Video
detect-traffic-light:
	python3 scripts/detect_traffic_light_depth.py $(if $(VIDEO),--video $(VIDEO),) $(if $(SAVE),--save,)

# Launch RViz2 inside Docker
rviz:
	docker compose exec aiformula_ws bash -c "source /opt/ros/humble/setup.bash && rviz2"




