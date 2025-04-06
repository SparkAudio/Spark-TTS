# Spark-TTS Docker User Guide

This directory contains all necessary files for running the Spark-TTS project in Docker.

<div align="center">
  <h3><a href="./Docker_README_CN.md">简体中文</a></h3>
</div>

## Directory structure

```
docker/
├── Dockerfile           # Docker image build file
├── docker-compose.yml   # Docker Compose configuration file
├── .dockerignore        # Docker build ignores files
├── download_models.py   # Download scripts for pre-trained models
├── start.sh             # Script to start Docker container
└── README.md            # This document
```

## Quick Start

1. Make sure you have Docker and Docker Compose installed
2. Run the startup script in the docker directory:
   ```bash
   sudo chmod +x start.sh
   ./start.sh
   ```
   
   This script will:
   - Automatically download the model, if not, download it
   - Build and start Docker container (background mode)


3. Open the browser and visit `http://localhost:12370` to use the WebUI.

## Commonly used commands

```bash
# Start the container (running backend)
docker compose up -d

# View container log
docker compose logs -f

# Stop the container
docker compose down

# Rebuild and start the container
docker compose up -d --build
```

## Things to note

1. The model file will be downloaded to the `pretrained_models` directory of the project root directory
2. The generated audio file is saved in the `example/results` directory of the project root directory
3. If you don't need GPU support, please comment out the GPU-related configuration in `docker-compose.yml`

## Troubleshooting

If you encounter problems, please check:

1. Is Docker and Docker Compose installed correctly
2. Is NVIDIA Docker Runtime correctly configured if using GPU
3. Is the network connection normal (you need to connect to the Internet to download the model)
4. Is port 12370 already occupied by other applications(you can change the port in `docker-compose.yml`)
