#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Spark-TTS API Client Example

This script demonstrates how to use the Spark-TTS API service through HTTP requests.
All features are integrated, and you can set the appropriate parameter combinations as needed.

Basic usage:
    python example_client.py --text "Text to synthesize"
    
Using reference audio for voice cloning:
    python example_client.py --text "Text to synthesize" --prompt_audio example/prompt_audio.wav
    
Using reference audio and text for more accurate voice cloning:
    python example_client.py --text "Text to synthesize" --prompt_audio example/prompt_audio.wav --prompt_text "Text content of the reference audio"
    
Using voice parameters for control:
    python example_client.py --text "Text to synthesize" --gender female --pitch high --speed moderate
    
Using both reference audio and voice parameters:
    python example_client.py --text "Text to synthesize" --prompt_audio example/prompt_audio.wav --gender male
    
Using API key:
    python example_client.py --text "Text to synthesize" --api_key YOUR_API_KEY

Output configuration:
    python example_client.py --text "Text to synthesize" --output_dir custom_outputs
    
Note: The server only accepts audio files in WAV format. If other formats are provided (such as MP3), the client will automatically convert them to WAV format.
"""

import os
import sys
import base64
import argparse
import requests
import tempfile
from urllib.parse import urljoin
import json


def convert_audio_to_wav(input_file, output_file=None):
    """Convert audio file to WAV format"""
    # If output file is not specified, create a temporary file
    if output_file is None:
        fd, output_file = tempfile.mkstemp(suffix=".wav")
        os.close(fd)
    
    # Check if input file exists
    if not os.path.exists(input_file):
        print(f"Error: Input file does not exist: {input_file}")
        return None
    
    # Check input file extension
    _, ext = os.path.splitext(input_file.lower())
    
    # If already in WAV format, copy directly
    if ext == '.wav':
        try:
            # Verify if it's a valid WAV file
            import soundfile as sf
            audio_data, sample_rate = sf.read(input_file)
            print(f"Input file is already in WAV format, sample rate: {sample_rate}Hz")
            
            # If output file is different from input file, copy the file
            if input_file != output_file:
                import shutil
                shutil.copy(input_file, output_file)
                
            return output_file
        except Exception as e:
            print(f"Warning: Input file has WAV extension but format is invalid: {str(e)}")
            # Continue trying to convert
    
    # Try different methods for conversion
    
    # Method 1: Try using ffmpeg (if installed on the system)
    try:
        import subprocess
        print(f"Attempting to convert using ffmpeg: {input_file}")
        result = subprocess.run(
            ["ffmpeg", "-i", input_file, "-ar", "16000", "-ac", "1", "-y", output_file],
            capture_output=True,
            text=True
        )
        if result.returncode == 0:
            print(f"Conversion with ffmpeg successful: {output_file}")
            return output_file
        else:
            print(f"ffmpeg conversion failed: {result.stderr}")
    except Exception as e:
        print(f"Conversion with ffmpeg failed: {str(e)}")
    
    # Method 2: Using librosa (supports multiple formats) - as fallback
    try:
        import librosa
        import soundfile as sf
        print(f"Loading audio using librosa: {input_file}")
        audio_data, sample_rate = librosa.load(input_file, sr=None)
        print(f"Converting to WAV format, sample rate: {sample_rate}Hz")
        sf.write(output_file, audio_data, sample_rate)
        print(f"Audio conversion successful: {output_file}")
        return output_file
    except ImportError:
        print("Warning: librosa library not installed, cannot use this method for conversion")
        print("Tip: Install librosa library to support more audio formats: pip install librosa")
    except Exception as e:
        print(f"Conversion with librosa failed: {str(e)}")
    
    print("Error: All conversion methods failed, unable to convert audio to WAV format")
    print("Please ensure ffmpeg is installed or install librosa library (pip install librosa)")
    return None


def read_audio_file(file_path):
    """Read audio file and convert to Base64 encoding"""
    # First ensure the file is in WAV format
    wav_file = convert_audio_to_wav(file_path)
    if not wav_file:
        raise ValueError(f"Unable to convert audio file to WAV format: {file_path}")
    
    # Read and encode WAV file
    with open(wav_file, "rb") as f:
        audio_data = f.read()
    
    # If it's a temporary file, delete it
    if wav_file != file_path:
        os.remove(wav_file)
        
    return base64.b64encode(audio_data).decode("utf-8")


def save_audio_file(base64_data, output_path):
    """Save Base64 encoded audio data as a file"""
    audio_data = base64.b64decode(base64_data)
    with open(output_path, "wb") as f:
        f.write(audio_data)


def tts_request(
    api_url,
    text,
    prompt_text=None,
    prompt_audio_path=None,
    prompt_audio_url=None,
    gender=None,
    pitch=None,
    speed=None,
    return_audio_data=True,
    api_key=None,
    output_dir="example_client_outputs",
    timeout=60,  # Add timeout parameter, default 60 seconds
):
    """
    Send TTS request to API
    
    Args:
        api_url: Base URL of the API service
        text: Text to synthesize
        prompt_text: Text content of the reference audio
        prompt_audio_path: Path to reference audio file
        prompt_audio_url: URL of reference audio
        gender: Voice gender
        pitch: Pitch
        speed: Speech rate
        return_audio_data: Whether to return Base64 encoded audio data
        api_key: API key
        output_dir: Output directory for client to locally save audio
        timeout: Request timeout in seconds
        
    Returns:
        Response dictionary or None
    """
    # Prepare URL
    endpoint = urljoin(api_url, "tts")
    
    # Check if URL pointing to the same service is used
    if prompt_audio_url:
        api_base = api_url.rstrip('/')
        if prompt_audio_url.startswith(api_base):
            print(f"Note: You are using a URL pointing to the same API service as reference audio: {prompt_audio_url}")
            print("The server will read the local file directly instead of downloading via HTTP")
    
    # Prepare request data
    payload = {"text": text, "return_audio_data": return_audio_data}
    
    # Add voice cloning parameters
    if prompt_text:
        payload["prompt_text"] = prompt_text
        
    if prompt_audio_path:
        try:
            # Read and encode audio file
            print(f"Reading audio file: {prompt_audio_path}")
            payload["prompt_audio_base64"] = read_audio_file(prompt_audio_path)
            print(f"Audio file encoding complete, size approximately {len(payload['prompt_audio_base64'])//1024} KB")
        except Exception as e:
            print(f"Failed to read audio file: {str(e)}")
            print("Please ensure the audio file exists and is in the correct format, or install librosa library to support more formats")
            return None
    elif prompt_audio_url:
        payload["prompt_audio_url"] = prompt_audio_url
        
    # Add voice creation parameters
    if gender:
        payload["gender"] = gender
    if pitch:
        payload["pitch"] = pitch
    if speed:
        payload["speed"] = speed
        
    # Prepare request headers
    headers = {"Content-Type": "application/json"}
    if api_key:
        headers["X-SPARKTTS-API-KEY"] = api_key
        
    # Send request
    print(f"Sending request to {endpoint}")
    print("Request processing, this may take some time...")
    
    try:
        response = requests.post(endpoint, json=payload, headers=headers, timeout=timeout)
        
        # Check response
        if response.status_code != 200:
            print(f"Error: {response.status_code} - {response.text}")
            
            # Special handling for audio format errors
            if response.status_code == 400 and "WAV format" in response.text:
                print("\nAudio format error: The server only accepts audio files in WAV format")
                print("The client attempted to automatically convert the audio format, but it may have failed")
                print("Suggestions:")
                print("1. Install librosa library: pip install librosa")
                print("2. Or manually convert the audio to WAV format before uploading")
                print("3. Or use ffmpeg to manually convert: ffmpeg -i input.mp3 -ar 16000 -ac 1 output.wav")
            
            return None
            
        # Parse response
        result = response.json()
        print("Request successful!")
        
        # Note: This example client saves audio files locally, which is different from the files saved by the server in the api/outputs directory:
        # - Server-side: Saves audio in the api/outputs directory, provides access via API URL
        # - Client-side: Saves a local copy of the audio in the example_client_outputs directory for local use
        
        # If Base64 audio was returned, save to file
        if return_audio_data and result.get("audio_base64"):
            # Ensure output directory exists
            os.makedirs(output_dir, exist_ok=True)
            
            # Save audio file
            output_path = os.path.join(output_dir, f"{result.get('file_id')}.wav")
            save_audio_file(result["audio_base64"], output_path)
            print(f"Client locally saved audio to: {output_path} (Note: The server also saves a copy in the api/outputs directory)")
            result["local_path"] = output_path
        elif result.get("audio_url"):
            print(f"Audio URL: {api_url.rstrip('/')}{result['audio_url']}")
        
        return result
    except requests.exceptions.Timeout:
        print(f"Request timeout, server processing time exceeded {timeout} seconds")
        print("This may be due to processing large audio files or high server load")
        print("You can try the following:")
        print("1. Use a smaller audio file")
        print("2. Increase timeout: --timeout 120")
        print("3. Check server logs for detailed errors")
        return None
    except requests.exceptions.ConnectionError:
        print("Connection error, unable to connect to server")
        print("Please ensure the API service is running and the port settings are correct")
        return None
    except Exception as e:
        print(f"Error sending request: {str(e)}")
        return None


def main():
    parser = argparse.ArgumentParser(description="Spark-TTS API Client Example")
    parser.add_argument("--api_url", default="http://localhost:7860/", help="Base URL of the API service")
    parser.add_argument("--api_key", default=None, help="API key")
    parser.add_argument("--text", default="Welcome to the Spark-TTS speech synthesis system.", help="Text to synthesize")
    parser.add_argument("--output_dir", default="example_client_outputs", help="Directory for client to locally save audio, separate from server-side storage directory")
    
    # Voice cloning parameters
    parser.add_argument("--prompt_text", default=None, help="Text content of the reference audio")
    parser.add_argument("--prompt_audio", default=None, help="Path to reference audio file")
    parser.add_argument("--prompt_audio_url", default=None, help="URL of reference audio")
    
    # Voice creation parameters
    parser.add_argument("--gender", choices=["male", "female"], default=None, help="Voice gender")
    parser.add_argument("--pitch", choices=["very_low", "low", "moderate", "high", "very_high"], default=None, help="Pitch")
    parser.add_argument("--speed", choices=["very_low", "low", "moderate", "high", "very_high"], default=None, help="Speech rate")
    
    # Other parameters
    parser.add_argument("--timeout", type=int, default=120, help="Request timeout in seconds")
    
    args = parser.parse_args()
    
    # Print audio save location note
    print(f"\nNote: The client will save a local copy of the audio file in the {args.output_dir} directory")
    print(f"At the same time, the server will also save the same audio file in the api/outputs directory\n")
    
    # Prepare feature description
    features = []
    if args.prompt_audio or args.prompt_audio_url:
        features.append("Voice Cloning")
    if args.gender or args.pitch or args.speed:
        features.append("Voice Creation")
    
    # Indicate which features are being used
    if features:
        print(f"Using features: {', '.join(features)}")
    else:
        print("Using default settings for TTS")
    
    # Make TTS request
    result = tts_request(
        api_url=args.api_url,
        text=args.text,
        prompt_text=args.prompt_text,
        prompt_audio_path=args.prompt_audio,
        prompt_audio_url=args.prompt_audio_url,
        gender=args.gender,
        pitch=args.pitch,
        speed=args.speed,
        api_key=args.api_key,
        output_dir=args.output_dir,
        timeout=args.timeout,
    )
    
    # Print result summary
    if result:
        print("\nResult Summary:")
        print(f"Text: {result['text']}")
        print(f"Duration: {result['duration']:.2f} seconds")
        print(f"Sample Rate: {result['sample_rate']} Hz")
        print(f"File ID: {result['file_id']}")
        print(f"Created At: {result['created_at']}")
        if "local_path" in result:
            print(f"Local File Path: {result['local_path']}")


if __name__ == "__main__":
    main() 