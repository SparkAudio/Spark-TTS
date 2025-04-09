#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Spark-TTS Web API
A FastAPI-based Spark-TTS Web API interface supporting speech synthesis, voice cloning, and voice creation features

Latest updates:
- Enhanced robustness of audio data processing, supporting multiple data types
- Improved error handling, providing more detailed log information
- Fixed data type mismatch issues
- Server only accepts audio in WAV format, other formats need to be converted on the client side
"""

import os
import base64
import shutil
import logging
import asyncio
import tempfile
import uuid
from datetime import datetime, timedelta
from typing import Optional, List, Dict, Any, Union, TypeVar
from pathlib import Path
from functools import lru_cache

import torch
import uvicorn
import requests
import soundfile as sf
from fastapi import FastAPI, Depends, HTTPException, Security, BackgroundTasks, UploadFile, File, Form, Query
from fastapi.security.api_key import APIKeyHeader, APIKey
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field, HttpUrl, validator
import numpy as np
from dotenv import load_dotenv

from cli.SparkTTS import SparkTTS

# Load .env file
project_root = Path(__file__).parent
env_file = project_root / '.env'
if env_file.exists():
    load_dotenv(env_file)
    logging.info(f"Environment variables file loaded: {env_file}")
else:
    logging.info(f"Environment variables file not found: {env_file}, using environment variables or default configuration")

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler()  # Ensure logs go to console
    ]
)
logger = logging.getLogger(__name__)

# Force logging level to INFO for this module
logger.setLevel(logging.INFO)

# Add a direct print of key configurations (will show even if logging is filtered)
def print_config_info():
    """Print configuration information directly to stdout, bypassing logging system"""
    settings = get_settings()
    print("\n" + "="*80)
    print("SPARK-TTS CONFIGURATION SUMMARY")
    print("="*80)
    
    # Print environment variables
    print("\nENVIRONMENT VARIABLES:")
    print(f"SPARK_TTS_DEFAULT_PROMPT_SPEECH = {os.getenv('SPARK_TTS_DEFAULT_PROMPT_SPEECH', 'not set')}")
    print(f"SPARK_TTS_MODEL_DIR = {os.getenv('SPARK_TTS_MODEL_DIR', 'not set')}")
    print(f"SPARK_TTS_OUTPUT_DIR = {os.getenv('SPARK_TTS_OUTPUT_DIR', 'not set')}")
    print(f"SPARK_TTS_DEVICE = {os.getenv('SPARK_TTS_DEVICE', 'not set')}")
    
    # Print calculated paths
    print("\nCALCULATED PATHS:")
    # Project root
    print(f"PROJECT_ROOT = {settings.PROJECT_ROOT}")
    print(f"Current directory = {os.getcwd()}")
    
    # Default prompt speech
    prompt_speech_path = settings.get_absolute_path(settings.DEFAULT_PROMPT_SPEECH)
    print(f"DEFAULT_PROMPT_SPEECH = {settings.DEFAULT_PROMPT_SPEECH}")
    print(f"  Absolute path = {prompt_speech_path}")
    print(f"  File exists = {os.path.exists(prompt_speech_path)}")
    
    # Model directory
    model_dir_path = settings.get_absolute_path(settings.MODEL_DIR)
    print(f"MODEL_DIR = {settings.MODEL_DIR}")
    print(f"  Absolute path = {model_dir_path}")
    print(f"  Directory exists = {os.path.exists(model_dir_path)}")
    
    # Output directory
    output_dir_path = settings.get_absolute_path(settings.OUTPUT_DIR)
    print(f"OUTPUT_DIR = {settings.OUTPUT_DIR}")
    print(f"  Absolute path = {output_dir_path}")
    print(f"  Directory exists = {os.path.exists(output_dir_path)}")
    
    print("="*80 + "\n")

# === Configuration Items ===
class Settings:
    # Service configuration
    API_PORT: int = int(os.getenv('SPARK_TTS_API_PORT', 7860))
    API_HOST: str = os.getenv('SPARK_TTS_API_HOST', '0.0.0.0')
    API_DEBUG: bool = os.getenv('SPARK_TTS_API_DEBUG', 'False').lower() == 'true'
    
    # Security configuration
    API_KEY_NAME: str = os.getenv('SPARK_TTS_API_KEY_NAME', 'X-SPARKTTS-API-KEY')
    API_KEY: Optional[str] = os.getenv('SPARK_TTS_API_KEY', None)
    
    # TTS model configuration
    MODEL_DIR: str = os.getenv('SPARK_TTS_MODEL_DIR', 'pretrained_models/Spark-TTS-0.5B')
    DEVICE: str = os.getenv('SPARK_TTS_DEVICE', 'gpu:0')
    
    # Default prompt audio and text
    DEFAULT_PROMPT_TEXT: str = os.getenv('SPARK_TTS_DEFAULT_PROMPT_TEXT', 
                                          "吃燕窝就选燕之屋，本节目由26年专注高品质燕窝的燕之屋冠名播出。豆奶牛奶换着喝，营养更均衡，本节目由豆本豆豆奶特约播出。")
    DEFAULT_PROMPT_SPEECH: str = os.getenv('SPARK_TTS_DEFAULT_PROMPT_SPEECH', 
                                           "example/prompt_audio.wav")
    logger.info(f"Environment variable SPARK_TTS_DEFAULT_PROMPT_SPEECH value: {os.getenv('SPARK_TTS_DEFAULT_PROMPT_SPEECH', 'not set')}")
    logger.info(f"Configured DEFAULT_PROMPT_SPEECH value: {DEFAULT_PROMPT_SPEECH}")
    
    # Output configuration
    OUTPUT_DIR: str = os.getenv('SPARK_TTS_OUTPUT_DIR', 'api/outputs')
    OUTPUT_URL_PREFIX: str = os.getenv('SPARK_TTS_OUTPUT_URL_PREFIX', '/outputs')
    
    # Cleanup configuration
    CLEANUP_INTERVAL: int = int(os.getenv('SPARK_TTS_CLEANUP_INTERVAL', 3600))  # seconds
    FILE_EXPIRY_TIME: int = int(os.getenv('SPARK_TTS_FILE_EXPIRY_TIME', 86400))  # seconds
    
    # Project root path
    @property
    def PROJECT_ROOT(self):
        # The parent directory of the api directory is the project root directory
        return os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    
    # Get absolute path
    def get_absolute_path(self, path, log=False):
        """Get absolute path, ensuring all relative paths are relative to the project root directory
        
        Args:
            path: The path to convert
            log: Whether to log the path conversion (default: False)
        
        Returns:
            Absolute path
        """
        if os.path.isabs(path):
            if log:
                logger.info(f"Path is already absolute: {path}")
            return path
        
        # If it's a path relative to the project root directory (starting with ../)
        if path.startswith("../"):
            abs_path = os.path.join(self.PROJECT_ROOT, path[3:])
            if log:
                logger.info(f"Converting relative path (../) to absolute: {path} -> {abs_path}")
            return abs_path
            
        # General relative path, considered relative to the project root directory
        # This ensures that the path resolution is consistent regardless of where the script is run from
        abs_path = os.path.join(self.PROJECT_ROOT, path)
        if log:
            logger.info(f"Converting relative path to absolute: {path} -> {abs_path}")
        return abs_path

@lru_cache
def get_settings():
    return Settings()

# === API Model ===
class TTSRequest(BaseModel):
    text: str = Field(..., description="Text to be synthesized")
    
    # Voice cloning parameters (all are optional)
    prompt_text: Optional[str] = Field(None, description="Text content of reference audio")
    prompt_audio_base64: Optional[str] = Field(None, description="Base64 encoded reference audio data")
    prompt_audio_url: Optional[HttpUrl] = Field(None, description="Reference audio URL")
    
    # Voice creation parameters (all are optional)
    gender: Optional[str] = Field(None, description="Voice gender (male/female)")
    pitch: Optional[str] = Field(None, description="Pitch (very_low/low/moderate/high/very_high)")
    speed: Optional[str] = Field(None, description="Speech speed (very_low/low/moderate/high/very_high)")
    
    # Advanced parameters
    temperature: float = Field(0.8, description="Sampling temperature")
    top_k: int = Field(50, description="Top K sampling")
    top_p: float = Field(0.95, description="Top P sampling")
    return_audio_data: bool = Field(False, description="Whether to include audio data in the response")

    @validator('prompt_audio_url')
    def validate_audio_sources(cls, v, values):
        if v is not None and values.get('prompt_audio_base64') is not None:
            raise ValueError("Cannot provide both prompt_audio_base64 and prompt_audio_url, please choose one method")
        return v
        
    @validator('text')
    def validate_text(cls, v):
        if not v or len(v.strip()) < 2:
            raise ValueError("Input text too short. Please provide at least 2 characters of text.")
        return v

class TTSResponse(BaseModel):
    text: str = Field(..., description="Input text")
    audio_url: Optional[str] = Field(None, description="Generated audio URL")
    audio_base64: Optional[str] = Field(None, description="Base64 encoded generated audio")
    duration: float = Field(..., description="Audio duration (seconds)")
    sample_rate: int = Field(..., description="Sample rate")
    file_id: str = Field(..., description="File ID")
    created_at: str = Field(..., description="Creation time")

# === Security ===
api_key_header = APIKeyHeader(name=get_settings().API_KEY_NAME, auto_error=False)

async def get_api_key(api_key: str = Security(api_key_header)):
    settings = get_settings()
    
    # If API key is not set, no validation is performed
    if not settings.API_KEY:
        return None
        
    if api_key == settings.API_KEY:
        return api_key
    
    raise HTTPException(
        status_code=403,
        detail="Invalid API key"
    )

# === Application Initialization ===
app = FastAPI(
    title="Spark-TTS API",
    description="A speech synthesis API based on Spark-TTS, supporting voice cloning and voice creation features",
    version="1.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# TTS model instance
tts_model = None

# Cleanup task
cleanup_task = None

# === Helper Functions ===
def initialize_model():
    """Initialize the TTS model
    
    Returns:
        SparkTTS: Initialized SparkTTS model
    """
    settings = get_settings()
    
    # Get model path
    model_dir = settings.get_absolute_path(settings.MODEL_DIR, log=True)
    logger.info(f"Initializing Spark-TTS model, path: {model_dir}")
    
    # Process device parameter
    device_param = settings.DEVICE.lower().strip()
    
    # Convert 'gpu' to appropriate device format for PyTorch
    if device_param == "gpu":
        device = "cuda" if torch.cuda.is_available() else "cpu"
        logger.info(f"Mapped 'gpu' to PyTorch device: {device}")
    elif device_param.startswith("gpu:"):
        gpu_id = device_param.split(":")[-1]
        if torch.cuda.is_available():
            device = f"cuda:{gpu_id}"
            logger.info(f"Using CUDA device {gpu_id}")
        else:
            device = "cpu"
            logger.warning("No CUDA available, falling back to CPU")
    else:
        # Use the device as-is
        device = device_param
    
    logger.info(f"Using inference device: {device}")
    
    # Initialize model
    try:
        model = SparkTTS(model_dir=model_dir, device=device)
        logger.info(f"Spark-TTS model initialization completed, actual used device: {model.device}")
        return model
    except Exception as e:
        logger.error(f"Model initialization failed: {str(e)}")
        raise

async def process_audio_source(request: TTSRequest) -> tuple:
    """Process audio source, return temporary file path and whether to clean up"""
    settings = get_settings()
    prompt_speech_path = None
    need_cleanup = False
    
    logger.info(f"Processing audio source - request.prompt_audio_base64 exists: {request.prompt_audio_base64 is not None}")
    logger.info(f"Processing audio source - request.prompt_audio_url exists: {request.prompt_audio_url is not None}")
    
    # If provided Base64 encoded audio
    if request.prompt_audio_base64:
        try:
            # Create temporary file
            fd, prompt_speech_path = tempfile.mkstemp(suffix=".wav")
            os.close(fd)
            logger.info(f"Created temporary file from Base64: {prompt_speech_path}")
            
            # Decode Base64 and write to temporary file
            audio_data = base64.b64decode(request.prompt_audio_base64)
            logger.info(f"Decoded audio data size: {len(audio_data)} bytes")
            
            # Write to temporary file
            with open(prompt_speech_path, "wb") as f:
                f.write(audio_data)
            
            # Verify whether it's a valid WAV file
            try:
                import soundfile as sf
                audio_data_sf, sample_rate = sf.read(prompt_speech_path)
                logger.info(f"Audio verification successful, sample rate: {sample_rate}")
            except Exception as e:
                logger.error(f"Invalid WAV audio file: {str(e)}")
                raise HTTPException(
                    status_code=400, 
                    detail="Provided audio is not a valid WAV format. Please convert audio to WAV format on the client side before uploading."
                )
                
            need_cleanup = True
            logger.info(f"Audio processing completed: {prompt_speech_path}")
            
        except HTTPException:
            # Re-raise HTTP exception
            raise
        except Exception as e:
            logger.error(f"Failed to process Base64 audio: {str(e)}", exc_info=True)
            raise HTTPException(status_code=400, detail=f"Invalid Base64 audio: {str(e)}")
    
    # If provided audio URL
    elif request.prompt_audio_url:
        try:
            # Create temporary file
            fd, prompt_speech_path = tempfile.mkstemp(suffix=".wav")
            os.close(fd)
            logger.info(f"Created temporary file from URL: {prompt_speech_path}")
            
            # Check if URL points to this service
            settings = get_settings()
            url_str = str(request.prompt_audio_url)
            server_host = f"http://{settings.API_HOST}:{settings.API_PORT}"
            local_urls = [
                f"http://localhost:{settings.API_PORT}",
                f"http://127.0.0.1:{settings.API_PORT}",
                server_host
            ]
            
            is_self_reference = False
            for local_url in local_urls:
                if url_str.startswith(local_url):
                    is_self_reference = True
                    # Extract file path
                    file_path = url_str.replace(f"{local_url}{settings.OUTPUT_URL_PREFIX}/", "")
                    logger.info(f"Detected self-reference, directly reading file: {file_path}")
                    
                    # Build local file path
                    local_file_path = os.path.join(settings.get_absolute_path(settings.OUTPUT_DIR, log=True), file_path)
                    logger.info(f"Local file path: {local_file_path}")
                    
                    if os.path.exists(local_file_path):
                        # Directly copy file instead of downloading via HTTP
                        import shutil
                        shutil.copy(local_file_path, prompt_speech_path)
                        logger.info(f"Direct copy of local file successful: {local_file_path} -> {prompt_speech_path}")
                    else:
                        logger.error(f"Local file does not exist: {local_file_path}")
                        raise HTTPException(status_code=404, detail=f"Local referenced file does not exist: {file_path}")
                    break
            
            # If not a self-reference, download URL audio and write to temporary file
            if not is_self_reference:
                # Download URL audio and write to temporary file
                logger.info(f"Downloading audio from URL: {request.prompt_audio_url}")
                response = requests.get(str(request.prompt_audio_url), stream=True)
                if response.status_code == 200:
                    with open(prompt_speech_path, "wb") as f:
                        for chunk in response.iter_content(chunk_size=8192):
                            f.write(chunk)
                    logger.info(f"URL audio written to temporary file successfully")
                else:
                    logger.error(f"Failed to download audio, HTTP status code: {response.status_code}")
                    raise HTTPException(status_code=400, detail=f"Failed to download audio: HTTP {response.status_code}")
            
            # Verify whether it's a valid WAV file
            try:
                import soundfile as sf
                audio_data_sf, sample_rate = sf.read(prompt_speech_path)
                logger.info(f"Audio verification successful, sample rate: {sample_rate}")
            except Exception as e:
                logger.error(f"Invalid WAV audio file: {str(e)}")
                raise HTTPException(
                    status_code=400, 
                    detail="Provided URL audio is not a valid WAV format. Please convert audio to WAV format on the client side before uploading."
                )
                
            need_cleanup = True
            
        except HTTPException:
            # Re-raise HTTP exception
            raise
        except Exception as e:
            logger.error(f"Failed to download audio: {str(e)}", exc_info=True)
            raise HTTPException(status_code=400, detail=f"Failed to download audio: {str(e)}")
    
    # If no audio is provided, use default audio
    else:
        # Get absolute path of default prompt audio
        prompt_speech_path = settings.get_absolute_path(settings.DEFAULT_PROMPT_SPEECH, log=True)
        logger.info(f"Using default prompt audio: {prompt_speech_path}")
        
        if not os.path.exists(prompt_speech_path):
            logger.warning(f"Default prompt audio does not exist: {prompt_speech_path}")
            # Try to find in different locations
            alt_paths = [
                os.path.join(settings.PROJECT_ROOT, "example/prompt_audio.wav"),
                os.path.join(os.getcwd(), "example/prompt_audio.wav"),
                os.path.join(os.path.dirname(os.path.abspath(__file__)), "../example/prompt_audio.wav")
            ]
            
            logger.info(f"Searching for alternative prompt audio files...")
            for path in alt_paths:
                logger.info(f"Checking alternative path: {path}")
                if os.path.exists(path):
                    logger.info(f"✅ Found alternative prompt audio: {path}")
                    settings.DEFAULT_PROMPT_SPEECH = path
                    logger.info(f"Updated DEFAULT_PROMPT_SPEECH value: {path}")
                    break
            else:
                logger.warning("❌ No alternative prompt audio files found in any location, service may not work properly")
        
        # Verify default audio is a valid WAV file
        try:
            import soundfile as sf
            audio_data_sf, sample_rate = sf.read(prompt_speech_path)
            logger.info(f"Default audio verification successful, sample rate: {sample_rate}")
        except Exception as e:
            logger.error(f"Default audio is not a valid WAV file: {str(e)}")
            raise HTTPException(status_code=500, detail="Default audio is not a valid WAV format")
        
        logger.info(f"Default prompt audio exists, size: {os.path.getsize(prompt_speech_path)} bytes")
    
    return prompt_speech_path, need_cleanup

def get_prompt_text(request: TTSRequest) -> str:
    """Get prompt text"""
    settings = get_settings()
    
    # Record input at call time
    logger.info(f"Getting prompt text - request.prompt_text: {request.prompt_text}")
    logger.info(f"Getting prompt text - Default prompt text: {settings.DEFAULT_PROMPT_TEXT}")
    
    if request.prompt_text:
        return request.prompt_text
    else:
        # Ensure default prompt text is not empty
        default_text = settings.DEFAULT_PROMPT_TEXT
        if not default_text or len(default_text.strip()) < 2:
            logger.warning("Default prompt text is empty or too short, using backup prompt text")
            default_text = "吃燕窝就选燕之屋，本节目由26年专注高品质燕窝的燕之屋冠名播出。豆奶牛奶换着喝，营养更均衡。"
        
        return default_text

AudioDataType = TypeVar('AudioDataType')

async def save_output_audio(audio_data, sample_rate: int = 16000) -> tuple:
    """Save output audio to file

    Args:
        audio_data: Audio data to save
        sample_rate: Sample rate of audio data

    Returns:
        tuple: (file_id, output_path, output_url, duration)
    """
    settings = get_settings()
    
    # Convert to numpy array if needed
    if isinstance(audio_data, torch.Tensor):
        audio_np = audio_data.cpu().numpy()
    else:
        audio_np = audio_data
    
    # Generate output file path
    file_id = str(uuid.uuid4())
    file_name = f"{file_id}.wav"
    
    # Ensure output directory exists
    output_dir = settings.get_absolute_path(settings.OUTPUT_DIR, log=True)
    os.makedirs(output_dir, exist_ok=True)
    
    # Create output file path
    output_path = os.path.join(output_dir, file_name)
    output_url = f"{settings.OUTPUT_URL_PREFIX}/{file_name}"
    
    # Calculate audio duration
    duration = len(audio_np) / sample_rate
    
    # Save audio to file
    try:
        sf.write(output_path, audio_np, sample_rate)
        logger.info(f"Saved audio to {output_path}, size: {os.path.getsize(output_path)} bytes")
        return file_id, output_path, output_url, duration
    except Exception as e:
        logger.error(f"Failed to save audio: {str(e)}")
        raise

def get_audio_base64(file_path: str) -> str:
    """Convert audio file to Base64 encoding"""
    with open(file_path, "rb") as f:
        audio_data = f.read()
    return base64.b64encode(audio_data).decode("utf-8")

async def cleanup_old_files():
    """Clean up expired output files"""
    settings = get_settings()
    
    while True:
        try:
            logger.info("Starting to clean up expired files...")
            now = datetime.now()
            expiry_time = now - timedelta(seconds=settings.FILE_EXPIRY_TIME)
            
            # Ensure output directory is absolute path (disable logging here to avoid duplication)
            output_dir = settings.get_absolute_path(settings.OUTPUT_DIR, log=False)
            
            # Ensure output directory exists
            if not os.path.exists(output_dir):
                logger.warning(f"Output directory does not exist: {output_dir}, skipping cleanup")
                await asyncio.sleep(settings.CLEANUP_INTERVAL)
                continue
                
            # Iterate through files in the output directory
            for filename in os.listdir(output_dir):
                file_path = os.path.join(output_dir, filename)
                
                # Check if the file is a regular file
                if not os.path.isfile(file_path):
                    continue
                    
                # Get file modification time
                file_mod_time = datetime.fromtimestamp(os.path.getmtime(file_path))
                
                # Delete expired files
                if file_mod_time < expiry_time:
                    try:
                        os.remove(file_path)
                        logger.info(f"Deleted expired file: {file_path}")
                    except Exception as e:
                        logger.error(f"Failed to delete file {file_path}: {str(e)}")
            
            deleted_count = 0
            logger.info(f"Cleanup complete, deleted {deleted_count} expired files")
            
        except Exception as e:
            logger.error(f"Error during cleanup: {str(e)}")
        
        # Wait for next cleanup interval
        await asyncio.sleep(settings.CLEANUP_INTERVAL)

# === API Endpoints ===
# Mount static files directory for audio output access
@app.post("/tts", response_model=TTSResponse)
async def text_to_speech(request: TTSRequest, api_key: APIKey = Depends(get_api_key)):
    """
    Text-to-speech endpoint supporting voice cloning and voice creation.
    
    This endpoint accepts text and optional parameters for voice cloning and/or voice creation,
    then generates audio using the Spark-TTS model.
    """
    settings = get_settings()
    logger.info(f"Received TTS request: {request.text[:100]}{'...' if len(request.text) > 100 else ''}")
    
    # Initialize model if not already initialized
    model = initialize_model()
    if model is None:
        raise HTTPException(status_code=500, detail="Failed to initialize TTS model")
    
    # Process audio source (for voice cloning)
    prompt_speech_path = None
    need_cleanup_audio = False
    
    if request.prompt_audio_base64 is not None or request.prompt_audio_url is not None:
        logger.info("Voice cloning mode detected")
        prompt_speech_path, need_cleanup_audio = await process_audio_source(request)
    else:
        logger.info("No voice cloning parameters provided, using default if available")
        # When no prompt audio is provided, we'll use the default if voice cloning is needed
        if any([param is not None for param in [request.gender, request.pitch, request.speed]]):
            logger.info("Voice creation mode detected")
        else:
            logger.info("Basic TTS mode, using default prompt")
            # In basic mode, we always use default prompt for better quality
            prompt_speech_path, need_cleanup_audio = await process_audio_source(request)
    
    # Get prompt text (for voice cloning)
    prompt_text = get_prompt_text(request)
    
    try:
        # Define the voice generation parameters
        tts_params = {}
        
        # Voice cloning parameters
        if prompt_speech_path:
            logger.info(f"Using prompt audio: {prompt_speech_path}")
            tts_params["prompt_audio"] = prompt_speech_path
            
        if prompt_text:
            logger.info(f"Using prompt text: {prompt_text[:100]}{'...' if len(prompt_text) > 100 else ''}")
            tts_params["prompt_text"] = prompt_text
            
        # Voice creation parameters
        if request.gender:
            logger.info(f"Setting gender: {request.gender}")
            tts_params["gender"] = request.gender
            
        if request.pitch:
            logger.info(f"Setting pitch: {request.pitch}")
            tts_params["pitch"] = request.pitch
            
        if request.speed:
            logger.info(f"Setting speed: {request.speed}")
            tts_params["speed"] = request.speed
        
        # Execute TTS inference
        logger.info(f"Starting TTS inference, text length: {len(request.text)}, text first 30 characters: {request.text[:30]}")
        try:
            # Add environment variable setting, which may help resolve CUDA errors
            os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
            logger.info("CUDA_LAUNCH_BLOCKING=1 set")
            
            # Convert prompt audio path to str type
            prompt_speech_path_str = str(prompt_speech_path)
            logger.info(f"Prompt audio path (str): {prompt_speech_path_str}")
            
            # Use asynchronous implementation with timeout handling
            async def run_inference():
                try:
                    if request.gender is not None:
                        # Voice creation mode
                        logger.info("Using voice creation mode")
                        return model.inference(
                            text=request.text,
                            gender=request.gender,
                            pitch=request.pitch or "moderate",
                            speed=request.speed or "moderate",
                            temperature=request.temperature,
                            top_k=request.top_k,
                            top_p=request.top_p
                        )
                    else:
                        # Voice cloning mode or basic mode
                        logger.info("Using voice cloning or basic mode")
                        logger.info(f"Parameter check - text: {request.text}")
                        logger.info(f"Parameter check - prompt_speech_path: {prompt_speech_path_str}")
                        logger.info(f"Parameter check - prompt_text: {prompt_text}")
                        
                        logger.info("Starting audio tokenization...")
                        # Use executor in thread pool to run this part
                        loop = asyncio.get_event_loop()
                        audio = await loop.run_in_executor(
                            None,
                            lambda: model.inference(
                                text=request.text,
                                prompt_speech_path=prompt_speech_path_str,
                                prompt_text=prompt_text,
                                temperature=request.temperature,
                                top_k=request.top_k,
                                top_p=request.top_p
                            )
                        )
                        logger.info("Audio synthesis completed")
                        return audio
                except Exception as e:
                    logger.error(f"Inference failed: {str(e)}", exc_info=True)
                    raise e
            
            # Set timeout time (120 seconds)
            try:
                logger.info("Starting TTS inference, setting timeout time to 120 seconds...")
                audio = await asyncio.wait_for(run_inference(), timeout=120)
                logger.info(f"TTS inference completed, audio shape: {audio.shape}")
            except asyncio.TimeoutError:
                logger.error("TTS inference timeout (120 seconds)")
                raise HTTPException(status_code=504, detail="TTS processing timeout, possibly due to reference audio too large or format incompatibility")
        except Exception as e:
            logger.error(f"TTS processing failed: {str(e)}", exc_info=True)
            
            # For debugging purposes, try to record CUDA device status
            try:
                logger.info(f"CUDA available: {torch.cuda.is_available()}")
                if torch.cuda.is_available():
                    logger.info(f"CUDA device count: {torch.cuda.device_count()}")
                    logger.info(f"Current CUDA device: {torch.cuda.current_device()}")
                    logger.info(f"CUDA device name: {torch.cuda.get_device_name(0)}")
                    logger.info(f"CUDA memory allocation: {torch.cuda.memory_allocated(0)}")
                    logger.info(f"CUDA memory cache: {torch.cuda.memory_reserved(0)}")
            except Exception as cuda_error:
                logger.error(f"Failed to get CUDA information: {str(cuda_error)}")
                
            raise HTTPException(status_code=500, detail=f"TTS processing failed: {str(e)}")
        
        # Clean up temporary audio file
        if need_cleanup_audio and prompt_speech_path and os.path.exists(prompt_speech_path):
            os.remove(prompt_speech_path)
        
        # Save output audio
        file_id, output_path, output_url, duration = await save_output_audio(audio)
        
        # Build response
        response = TTSResponse(
            text=request.text,
            audio_url=output_url,
            audio_base64=get_audio_base64(output_path) if request.return_audio_data else None,
            duration=duration,
            sample_rate=16000,
            file_id=file_id,
            created_at=datetime.now().isoformat()
        )
        
        return response
        
    except Exception as e:
        logger.error(f"TTS processing failed: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"TTS processing failed: {str(e)}")

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    global tts_model
    settings = get_settings()
    
    # Get device information
    device_info = {
        "configured": settings.DEVICE,
        "actual": str(tts_model.device) if tts_model is not None else "Not initialized"
    }
    
    return {
        "status": "ok", 
        "timestamp": datetime.now().isoformat(),
        "device": device_info,
        "model_loaded": tts_model is not None
    }

# === Application startup and shutdown events ===
@app.on_event("startup")
async def startup_event():
    """Initialize the TTS model on startup"""
    # Load environment variables
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    
    global cleanup_task
    settings = get_settings()
    
    # Print essential configuration information (bypassing logging system)
    # This direct print ensures critical config is always visible even if logging is filtered
    print_config_info()
    
    # Initialize model
    try:
        model = initialize_model()
    except Exception as e:
        logger.error(f"Failed to initialize model: {str(e)}")
        model = None
    
    # Start scheduled cleanup task
    cleanup_task = asyncio.create_task(cleanup_old_files())
    
    # Mount static files directory
    try:
        # Ensure output directory is absolute path (log only once during startup)
        output_dir = settings.get_absolute_path(settings.OUTPUT_DIR, log=True)
        os.makedirs(output_dir, exist_ok=True)
        logger.info(f"Using output directory: {output_dir}")
        
        # Remove URL prefix leading slash, if any
        static_url_path = settings.OUTPUT_URL_PREFIX
        if static_url_path.startswith('/'):
            static_url_path = static_url_path[1:]
            
        # Mount static files directory
        app.mount(f"/{static_url_path}", StaticFiles(directory=output_dir), name="audio_files")
        logger.info(f"Mounted static files directory: {output_dir} to /{static_url_path}")
    except Exception as e:
        logger.error(f"Mounting static files directory failed: {str(e)}")
    
    logger.info("Spark-TTS API service started successfully")

@app.on_event("shutdown")
async def shutdown_event():
    global cleanup_task
    
    # Cancel cleanup task
    if cleanup_task:
        cleanup_task.cancel()
        try:
            await cleanup_task
        except asyncio.CancelledError:
            pass
    
    logger.info("Spark-TTS API service shut down")

# === Main program ===
if __name__ == "__main__":
    settings = get_settings()
    uvicorn.run(
        app, 
        host=settings.API_HOST, 
        port=settings.API_PORT,
        reload=settings.API_DEBUG
    ) 