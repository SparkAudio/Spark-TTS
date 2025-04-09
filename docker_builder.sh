#!/bin/bash

# Spark-TTS Docker Image Builder
# This script builds different versions of the Spark-TTS Docker image

# Set colors for output
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m' # No Color

# Get the absolute path of the script
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"
cd "$SCRIPT_DIR"

# Define image names
IMAGE_NAME="spark-tts"
BASE_TAG="latest"
FULL_TAG="latest-full"
LITE_TAG="latest-lite"

# Print header
echo -e "${GREEN}====================================${NC}"
echo -e "${GREEN}   Spark-TTS Docker Image Builder   ${NC}"
echo -e "${GREEN}====================================${NC}"
echo

# Check if Docker is installed
if ! command -v docker &> /dev/null; then
    echo -e "${RED}Error: Docker is not installed or not in PATH${NC}"
    exit 1
fi

# Check if pretrained_models directory exists
if [ ! -d "./pretrained_models" ]; then
    echo -e "${YELLOW}Warning: pretrained_models directory not found${NC}"
    echo -e "${YELLOW}Models will not be included in the 'full' image${NC}"
    echo -e "${YELLOW}You can download models later or mount them when running the container${NC}"
    read -p "Do you want to continue? (y/n): " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        echo -e "${RED}Build canceled${NC}"
        exit 1
    fi
fi

# Build lite version (without models)
echo -e "${GREEN}Building ${IMAGE_NAME}:${LITE_TAG} (without models)...${NC}"
docker build -t ${IMAGE_NAME}:${LITE_TAG} .

if [ $? -ne 0 ]; then
    echo -e "${RED}Failed to build ${IMAGE_NAME}:${LITE_TAG}${NC}"
    exit 1
fi

# Set as latest tag
echo -e "${GREEN}Tagging ${IMAGE_NAME}:${LITE_TAG} as ${IMAGE_NAME}:${BASE_TAG}...${NC}"
docker tag ${IMAGE_NAME}:${LITE_TAG} ${IMAGE_NAME}:${BASE_TAG}

# Build full version (with models)
echo -e "${GREEN}Building ${IMAGE_NAME}:${FULL_TAG} (with models)...${NC}"
docker build --build-arg INCLUDE_MODELS=true -t ${IMAGE_NAME}:${FULL_TAG} .

if [ $? -ne 0 ]; then
    echo -e "${RED}Failed to build ${IMAGE_NAME}:${FULL_TAG}${NC}"
    echo -e "${YELLOW}Note: The lite version was built successfully and is available${NC}"
    exit 1
fi

# Summary
echo
echo -e "${GREEN}====================================${NC}"
echo -e "${GREEN}    Build Completed Successfully    ${NC}"
echo -e "${GREEN}====================================${NC}"
echo
echo -e "Image tags created:"
echo -e "  - ${IMAGE_NAME}:${BASE_TAG} (alias of ${LITE_TAG})"
echo -e "  - ${IMAGE_NAME}:${LITE_TAG} (without models)"
echo -e "  - ${IMAGE_NAME}:${FULL_TAG} (with models)"
echo
echo -e "To run API (default):"
echo -e "  docker run -p 7860:7860 --gpus all ${IMAGE_NAME}:${FULL_TAG}"
echo
echo -e "To run WebUI:"
echo -e "  docker run -p 7860:7860 --gpus all -e SERVICE_TYPE=webui ${IMAGE_NAME}:${FULL_TAG}"
echo
echo -e "To use the lite version, you must mount the models directory:"
echo -e "  docker run -p 7860:7860 --gpus all -v /local/path/to/models:/app/pretrained_models ${IMAGE_NAME}:${LITE_TAG}"
echo 