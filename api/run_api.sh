#!/bin/bash

# Get the absolute path of the script
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"

# Get the project root directory
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

# Set essential environment variables
export PYTHONPATH="$PROJECT_ROOT:$PYTHONPATH"
export CUDA_LAUNCH_BLOCKING=1

# 1. Load environment variables from .env file (lowest priority)
ENV_FILE="$SCRIPT_DIR/.env"
if [ -f "$ENV_FILE" ]; then
  echo "Loading environment variables from: $ENV_FILE"
  # Export environment variables from .env file
  while IFS='=' read -r key value || [ -n "$key" ]; do
    # Skip comment lines and empty lines
    [[ $key == \#* ]] && continue
    [[ -z "$key" ]] && continue
    
    # Remove surrounding quotes
    value=$(echo "$value" | sed -e 's/^"//' -e 's/"$//' -e "s/^'//" -e "s/'$//")
    
    # Only export if environment variable is not already set
    if [ -z "${!key}" ]; then
      export "$key=$value"
    fi
  done < "$ENV_FILE"
else
  echo "No environment file detected, using system environment variables or defaults"
fi

# 2. Set defaults or read from environment variables (middle priority)
# Using environment variables if available, otherwise use defaults
HOST="${SPARK_TTS_API_HOST:-0.0.0.0}"
PORT="${SPARK_TTS_API_PORT:-7860}"
DEBUG="${SPARK_TTS_API_DEBUG:-false}"
RELOAD=false

# 3. Process command line arguments (highest priority)
while [[ $# -gt 0 ]]; do
  case $1 in
    --host)
      HOST="$2"
      shift 2
      ;;
    --port)
      PORT="$2"
      shift 2
      ;;
    --debug)
      DEBUG=true
      RELOAD=true
      shift
      ;;
    --reload)
      RELOAD=true
      shift
      ;;
    *)
      echo "Unknown option: $1"
      exit 1
      ;;
  esac
done

# Convert string boolean to actual boolean for DEBUG if needed
if [[ "$DEBUG" == "true" || "$DEBUG" == "True" || "$DEBUG" == "TRUE" || "$DEBUG" == "1" ]]; then
  RELOAD=true
fi

# Create output directory if it doesn't exist
OUTPUT_DIR_PATH="${SPARK_TTS_OUTPUT_DIR:-api/outputs}"
if [[ "$OUTPUT_DIR_PATH" = /* ]]; then
  # Absolute path
  FINAL_OUTPUT_DIR="$OUTPUT_DIR_PATH"
else
  # Relative path, convert to absolute path
  FINAL_OUTPUT_DIR="$PROJECT_ROOT/$OUTPUT_DIR_PATH"
fi
mkdir -p "$FINAL_OUTPUT_DIR"

# Start API service
cd "$PROJECT_ROOT"
echo "Starting Spark-TTS API service..."
echo "Host: $HOST, Port: $PORT, Debug: $DEBUG, Reload: $RELOAD"

# Set RELOAD parameter
if [ "$RELOAD" = true ]; then
  RELOAD_ARG="--reload"
else
  RELOAD_ARG=""
fi

# Start API service
python -m uvicorn api.main:app --host "$HOST" --port "$PORT" $RELOAD_ARG

# Check exit status
if [ $? -ne 0 ]; then
  echo "Failed to start API service!"
  exit 1
fi

echo "API service has stopped" 