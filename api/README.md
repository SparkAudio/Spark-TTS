# Spark-TTS API

This is a Web API interface based on FastAPI for accessing the functionality of the Spark-TTS speech synthesis model. Compared to the existing WebUI interface, this API provides more flexible feature selection and supports both voice cloning and voice creation features.

## Features

- Supports basic text-to-speech synthesis
- Supports voice cloning based on reference audio
- Supports voice creation based on parameter control
- Supports simultaneous use of voice cloning and voice creation features
- Supports multiple audio input methods: Base64, URL, default audio
- Supports multiple audio output methods: Base64, URL access
- Supports API key authentication (optional)
- Automatically cleans up expired audio files
- Flexible configuration, supports settings through environment variables or .env file

## Getting Started

### Configuring Environment Variables

The API supports configuring environment variables in two ways:

1. **Using a .env file (recommended)**:
   ```bash
   # Copy the example configuration file
   cp api/.env.example api/.env
   
   # Edit the configuration file
   nano api/.env
   ```

2. **Setting environment variables directly**:
   ```bash
   export SPARK_TTS_API_PORT=8080
   export SPARK_TTS_DEVICE=1
   # Other environment variables...
   ```

### Starting the API Service

Use the provided script to start the API service:

```bash
chmod +x api/run_api.sh
./api/run_api.sh
```

If you used a .env file, the script will automatically load the configurations from it. You can also override the settings in the .env file with command line arguments:

```bash
./api/run_api.sh --port 8080 --device 1 --model_dir /path/to/model --debug
```

### Custom Startup Parameters

You can customize the behavior of the service with the following parameters:

```bash
./api/run_api.sh --port 8080 --device 1 --model_dir /path/to/model --debug --env /path/to/env/file
```

Parameter descriptions:
- `--port`: Port for the service to listen on (default: 7860)
- `--device`: GPU device ID to use (default: 0)
- `--model_dir`: Model directory path (default: pretrained_models/Spark-TTS-0.5B)
- `--debug`: Enable debug mode
- `--env`: Specify a custom .env file path (default: api/.env)

## API Endpoints

### 1. Text to Speech (POST /tts)

Convert text to speech, supporting voice cloning and voice creation.

**Request Parameters**:

```json
{
  "text": "Text to synthesize",
  
  // Voice cloning parameters (optional)
  "prompt_text": "Text content of the reference audio",
  "prompt_audio_base64": "Base64 encoded reference audio",
  "prompt_audio_url": "URL of the reference audio",
  
  // Voice creation parameters (optional)
  "gender": "male or female",
  "pitch": "very_low, low, moderate, high, or very_high",
  "speed": "very_low, low, moderate, high, or very_high",
  
  // Other parameters
  "temperature": 0.8,
  "top_k": 50,
  "top_p": 0.95,
  "return_audio_data": true
}
```

**Note**: `prompt_audio_base64` and `prompt_audio_url` are mutually exclusive parameters and cannot be provided simultaneously.

**Response**:

```json
{
  "text": "Input text",
  "audio_url": "/outputs/file_id.wav",
  "audio_base64": "Base64 encoded audio (when return_audio_data=true)",
  "duration": 3.5,
  "sample_rate": 16000,
  "file_id": "unique_file_id",
  "created_at": "2023-05-20T12:34:56.789"
}
```

### 2. Get Audio File (GET /outputs/{file_id})

Retrieve the generated audio file using the file ID.

**Request**:

```
GET /outputs/{file_id}
```

**Response**:
Audio file (WAV format)

### 3. Health Check (GET /health)

Check if the API service is running normally.

**Request**:

```
GET /health
```

**Response**:

```json
{
  "status": "ok",
  "timestamp": "2023-05-20T12:34:56.789",
  "device": {
    "configured": "gpu:0",
    "actual": "cuda:0"
  },
  "model_loaded": true
}
```

## Example Client

An example client script is provided to demonstrate how to use the API:

```bash
# Note: The example client requires librosa, which is not in requirements.txt
# Install it before running the client:
pip install librosa

# Basic usage
python api/example_client.py --text "This is a test"
```

You can use different parameter combinations for different features:

- **For voice cloning** (using reference audio):
```bash
python api/example_client.py --text "This is an example of voice cloning" --prompt_audio example/prompt_audio.wav
```

- **For voice creation** (using control parameters):
```bash
python api/example_client.py --text "This is an example of voice creation" --gender female --pitch high --speed moderate
```

- **Combined features** (both voice cloning and creation):
```bash
python api/example_client.py --text "This is an example of combined features" --prompt_audio example/prompt_audio.wav --gender male --pitch low
```

- **Using with API key**:
```bash
python api/example_client.py --text "This is a test" --api_key YOUR_API_KEY
```

## Environment Variable Configuration

The API service can be configured using the following environment variables:

| Environment Variable | Description | Default Value |
|----------|------|--------|
| SPARK_TTS_API_PORT | API service port | 7860 |
| SPARK_TTS_API_HOST | API service host | 0.0.0.0 |
| SPARK_TTS_API_DEBUG | Whether to enable debug mode | False |
| SPARK_TTS_API_KEY | API key (authentication not enabled if not set) | None |
| SPARK_TTS_API_KEY_NAME | API key request header name | X-SPARKTTS-API-KEY |
| SPARK_TTS_MODEL_DIR | Model directory path | pretrained_models/Spark-TTS-0.5B |
| SPARK_TTS_DEVICE | GPU device ID | gpu:0 |
| SPARK_TTS_DEFAULT_PROMPT_TEXT | Default reference text | "吃燕窝就选燕之屋..." |
| SPARK_TTS_DEFAULT_PROMPT_SPEECH | Default reference audio path | example/prompt_audio.wav |
| SPARK_TTS_OUTPUT_DIR | Output audio file directory | api/outputs |
| SPARK_TTS_OUTPUT_URL_PREFIX | Output audio URL prefix | /outputs |
| SPARK_TTS_CLEANUP_INTERVAL | Cleanup task interval (seconds) | 3600 |
| SPARK_TTS_FILE_EXPIRY_TIME | File expiration time (seconds) | 86400 |

## Docker Support

The API design considers operation in a Docker environment and can be flexibly configured through environment variables or by mounting the api/.env file. Dedicated Docker support will be provided in the future.

Example Docker mount command:
```bash
docker run -p 7860:7860 -v /local/path/api/.env:/app/api/.env -v /local/path/pretrained_models:/app/pretrained_models spark-tts:latest
```

## Notes

- Please ensure that the model files have been correctly downloaded and placed in the specified directory.
- If API key authentication is enabled, all requests must include the correct API key header.
- Generated audio files will be automatically deleted after the set expiration time, default is 24 hours.
- Static file service has been configured, and generated audio files can be accessed directly via URL.
- **The server only accepts audio files in WAV format**. If you need to use other formats (such as MP3), please convert them to WAV format on the client side before uploading. The example client includes automatic conversion functionality.
- When using the `prompt_audio_url` parameter to point to an audio file on the same server (such as `http://localhost:7860/outputs/xxx.wav`), the server will read the local file directly rather than downloading it via HTTP to avoid circular reference issues. 