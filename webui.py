# Copyright (c) 2025 SparkAudio
#               2025 Xinsheng Wang (w.xinshawn@gmail.com)
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os
import torch
import soundfile as sf
import logging
import argparse
import gradio as gr
import subprocess
import re
import glob
from datetime import datetime
from pathlib import Path

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('spark_tts.log')
    ]
)

from cli.SparkTTS import SparkTTS
from sparktts.utils.token_parser import LEVELS_MAP_UI

# Create a custom temp directory in the same folder as the script
TEMP_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "temp")
os.makedirs(TEMP_DIR, exist_ok=True)

# Global variable to store the model instance
model_instance = None

# Define a GenerationConfig class to match the expected interface
class GenerationConfig:
    def __init__(self, temperature=0.8, top_p=0.95, top_k=50):
        self.temperature = temperature
        self.top_p = top_p
        self.top_k = top_k

def initialize_model(model_dir="pretrained_models/Spark-TTS-0.5B", device=0):
    """Load the model once at the beginning."""
    global model_instance
    
    try:
        # Check if model directory exists
        model_path = os.path.abspath(model_dir)
        if not os.path.exists(model_path):
            logging.error(f"Model directory not found: {model_path}")
            return False
            
        # Check if LLM subdirectory exists
        llm_path = os.path.join(model_path, "LLM")
        if not os.path.exists(llm_path):
            logging.error(f"LLM directory not found: {llm_path}")
            return False
            
        # Setup device
        if device >= 0 and torch.cuda.is_available():
            device = f"cuda:{device}"
        else:
            device = "cpu"
        
        logging.info(f"Initializing model from {model_path} on {device}...")
        model_instance = SparkTTS(Path(model_path), torch.device(device))
        logging.info(f"Model loaded from {model_path} on {device}")
        return True
    except Exception as e:
        import traceback
        logging.error(f"Failed to initialize model: {e}")
        logging.error(traceback.format_exc())
        return False


def run_tts(
    text,
    model,
    prompt_text=None,
    prompt_speech=None,
    gender=None,
    pitch=None,
    speed=None,
    temperature=0.8,
    top_p=0.95,
    top_k=50,
    seed=None,
    save_dir=None,
    progress=None
):
    """Perform TTS inference and save the generated audio."""
    try:
        # Use our custom temp directory
        if save_dir is None:
            save_dir = TEMP_DIR
            
        logging.info(f"Saving audio to: {save_dir}")

        if prompt_text is not None:
            prompt_text = None if len(prompt_text) <= 1 else prompt_text

        # Ensure the save directory exists
        os.makedirs(save_dir, exist_ok=True)

        # Generate unique filename using timestamp and a preview of the text
        timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
        # Use the first few words of text for the filename
        text_preview = re.sub(r'[^\w\s]', '', text)[:20].strip().replace(' ', '_')
        if not text_preview:
            text_preview = "tts_output"
            
        # Add type of generation to filename
        if prompt_speech is not None:
            gen_type = "clone"
        elif gender is not None:
            gen_type = f"{gender}_{pitch}_{speed}"
        else:
            gen_type = "tts"
            
        save_path = os.path.join(save_dir, f"{timestamp}_{gen_type}_{text_preview}.wav")

        logging.info("Starting inference...")

        # Set seed for reproducibility if provided
        if seed is not None:
            try:
                seed = int(seed)
                torch.manual_seed(seed)
                if torch.cuda.is_available():
                    torch.cuda.manual_seed_all(seed)
            except Exception as e:
                logging.warning(f"Failed to set seed: {e}")
        
        # Cap temperature to avoid generating noise/silence
        temperature = min(temperature, 1.2)
                
        # Perform inference and save the output audio
        with torch.no_grad():
            wav = model.inference(
                text,
                prompt_speech,
                prompt_text,
                gender,
                pitch,
                speed,
                temperature=temperature,
                top_p=top_p,
                top_k=top_k,
            )
                
            # Trim silence from the end of the audio
            # Find the last non-zero sample
            threshold = 0.01  # Adjust as needed
            last_idx = len(wav) - 1
            while last_idx > 0 and abs(wav[last_idx]) < threshold:
                last_idx -= 1
                
            # Add a small buffer to avoid cutting off speech
            buffer_samples = int(0.2 * 16000)  # 200ms at 16kHz
            last_idx = min(len(wav) - 1, last_idx + buffer_samples)
            
            # Trim the audio
            wav = wav[:last_idx + 1]

            sf.write(save_path, wav, samplerate=16000)

        logging.info(f"Audio saved at: {save_path}")
        return save_path
    except Exception as e:
        logging.error(f"Error in run_tts: {e}")
        return None


def voice_clone(text, prompt_text, prompt_wav_upload, prompt_wav_record, temperature=0.8, top_p=0.95, top_k=50, seed=None):
    """Voice cloning function."""
    if not text:
        return "Please enter some text to convert to speech."

    # Use uploaded or recorded audio
    prompt_wav = prompt_wav_upload or prompt_wav_record
    if not prompt_wav:
        return "Please upload or record audio for voice cloning."

    try:
        # Clean and prepare prompt text 
        if prompt_text and len(prompt_text.strip()) > 0:
            prompt_text = prompt_text.strip()
        else:
            prompt_text = None

        # Convert parameters to appropriate types
        try:
            temperature = float(temperature) if temperature is not None else 0.8
            top_p = float(top_p) if top_p is not None else 0.95
            top_k = int(top_k) if top_k is not None else 50
            seed = int(seed) if seed and seed.strip() else None
        except (ValueError, TypeError) as e:
            logging.warning(f"Parameter conversion error: {e}")
            temperature = 0.8
            top_p = 0.95
            top_k = 50
            seed = None

        # Ensure temperature is in a reasonable range
        temperature = max(0.1, min(temperature, 1.2))
        
        # Generate audio
        output_path = run_tts(
            text=text,
            model=model_instance,
            prompt_text=prompt_text,
            prompt_speech=prompt_wav,
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            seed=seed
        )

        if output_path:
            return output_path
        else:
            return "Failed to generate audio. Check logs for details."
    except Exception as e:
        logging.error(f"Voice cloning error: {e}")
        return f"Error: {str(e)}"


def voice_creation(text, gender, pitch, speed, temperature=0.8, top_p=0.95, top_k=50, seed=None):
    """Voice creation function."""
    if not text:
        return "Please enter some text to convert to speech."

    try:
        # Map pitch and speed values from UI to internal values
        try:
            pitch_val = LEVELS_MAP_UI[int(pitch)] if pitch is not None else "moderate"
            speed_val = LEVELS_MAP_UI[int(speed)] if speed is not None else "moderate"
        except (ValueError, KeyError) as e:
            logging.warning(f"Failed to map pitch/speed: {e}")
            pitch_val = "moderate"
            speed_val = "moderate"
            
        # Convert parameters to appropriate types
        try:
            temperature = float(temperature) if temperature is not None else 0.8
            top_p = float(top_p) if top_p is not None else 0.95
            top_k = int(top_k) if top_k is not None else 50
            seed = int(seed) if seed and seed.strip() else None
        except (ValueError, TypeError) as e:
            logging.warning(f"Parameter conversion error: {e}")
            temperature = 0.8
            top_p = 0.95
            top_k = 50
            seed = None

        # Ensure temperature is in a reasonable range
        temperature = max(0.1, min(temperature, 1.2))
        
        # Generate audio
        output_path = run_tts(
            text=text,
            model=model_instance,
            gender=gender,
            pitch=pitch_val,
            speed=speed_val,
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            seed=seed
        )

        if output_path:
            return output_path
        else:
            return "Failed to generate audio. Check logs for details."
    except Exception as e:
        logging.error(f"Voice creation error: {e}")
        return f"Error: {str(e)}"


def build_ui(model_dir, device=0):
    """Build the gradio UI."""
    global model_instance
    
    if model_instance is None:
        if not initialize_model(model_dir, device):
            raise RuntimeError("Failed to initialize model")
    
    # Create Gradio blocks
    demo = gr.Blocks(
        css="""
            #output-audio-clone, #output-audio-creation {
                margin-top: 10px;
            }
            .gradio-container {
                max-width: 960px !important;
                margin: auto;
            }
        """
    )

    # Custom CSS for a more modern look
    custom_css = """
    .gradio-container {
        font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
    }
    .main-header {
        background: linear-gradient(90deg, #5e72e4 0%, #825ee4 100%);
        color: white !important; /* Force white text regardless of theme */
        padding: 20px;
        border-radius: 12px;
        margin-bottom: 20px;
        box-shadow: 0 4px 6px rgba(50, 50, 93, 0.11), 0 1px 3px rgba(0, 0, 0, 0.08);
        width: 100%;
        text-align: center;
    }
    .main-header h1, .main-header p {
        color: white !important;
        margin: 0;
    }
    .main-header p {
        margin-top: 10px;
    }
    .tabs > .tab-nav {
        gap: 8px;
        margin-bottom: 20px;
    }
    .tabs > .tab-nav button {
        border: none;
        border-radius: 8px;
        padding: 10px 20px;
        font-weight: 600;
        transition: all 0.3s ease;
    }
    .tabs > .tab-nav button.selected {
        background-color: #5e72e4;
        color: white;
    }
    .block {
        border-radius: 12px;
        padding: 20px;
        box-shadow: 0 2px 4px rgba(0, 0, 0, 0.05);
        margin-bottom: 15px;
    }
    button.primary {
        background: linear-gradient(90deg, #5e72e4 0%, #825ee4 100%) !important;
        border: none !important;
        border-radius: 8px !important;
        color: white !important;
        font-weight: bold !important;
        padding: 12px 24px !important;
        transition: all 0.3s ease !important;
    }
    button.primary:hover {
        transform: translateY(-2px);
        box-shadow: 0 7px 14px rgba(50, 50, 93, 0.1), 0 3px 6px rgba(0, 0, 0, 0.08);
    }
    .output-container {
        border-radius: 12px;
        padding: 20px;
        box-shadow: 0 2px 4px rgba(0, 0, 0, 0.05);
    }
    .slider {
        margin: 10px 0;
    }
    """

    # Create a customized theme
    theme = gr.themes.Soft(
        primary_hue="indigo",
        secondary_hue="purple",
    ).set(
        body_background_fill="white",
        block_background_fill="*neutral_50",
        block_label_background_fill="*primary_100",
        button_primary_background_fill="*primary_500",
        button_primary_background_fill_hover="*primary_600",
        button_primary_text_color="white",
        slider_color="*primary_500",
        slider_color_dark="*primary_600"
    )

    with demo:
        # Modern header
        gr.HTML('<div class="main-header"><h1>Spark-TTS</h1><p>Advanced voice synthesis powered by SparkAudio</p></div>')
        
        with gr.Tabs(elem_classes="tab-nav"):
            # Voice Clone Tab
            with gr.TabItem("Voice Clone"):
                gr.Markdown("### Clone a voice from reference audio")

                # Two-column layout for inputs/outputs and controls
                with gr.Row():
                    # Inputs and outputs in the left column
                    with gr.Column():
                        # Reference audio section
                        with gr.Column(elem_classes="block"):
                            gr.Markdown("### Reference Audio")
                            
                            with gr.Row():
                                with gr.Column():
                                    prompt_wav_upload = gr.Audio(
                                        sources="upload",
                                        type="filepath",
                                        label="Upload reference audio (16kHz or higher)",
                                        elem_id="voice-upload"
                                    )
                                
                                with gr.Column():
                                    prompt_wav_record = gr.Audio(
                                        sources="microphone",
                                        type="filepath",
                                        label="Record your voice",
                                        elem_id="voice-record"
                                    )
                        
                        # Text input section
                        with gr.Column(elem_classes="block"):
                            gr.Markdown("### Text Input")
                            
                            text_input = gr.Textbox(
                                label="Text to Synthesize",
                                lines=3,
                                placeholder="Enter the text you want to convert to speech...",
                                elem_id="synthesis-text"
                            )
                            
                            prompt_text_input = gr.Textbox(
                                label="Reference Audio Transcript (Optional)",
                                lines=3,
                                placeholder="Enter the transcript of your reference audio for better results...",
                                elem_id="reference-text"
                            )
                        
                        # Output section
                        with gr.Column(elem_classes="block"):
                            gr.Markdown("### Output")
                            
                            audio_output_clone = gr.Audio(
                                label="Generated Audio"
                            )
                            
                            generate_button_clone = gr.Button("Generate Voice", elem_classes="primary")
                    
                    # Controls in the right column
                    with gr.Column():
                        # Generation parameters controls
                        with gr.Column(elem_classes="block"):
                            gr.Markdown("### Generation Parameters")
                            
                            with gr.Row():
                                with gr.Column():
                                    temperature = gr.Slider(
                                        minimum=0.1,
                                        maximum=1.2,
                                        step=0.05,
                                        value=0.7,
                                        label="Temperature",
                                        info="Controls randomness (higher = more diverse outputs). Lower values produce more consistent results with less silence.",
                                        elem_id="temperature-slider"
                                    )
                                
                            with gr.Row():
                                with gr.Column():
                                    top_p = gr.Slider(
                                        minimum=0.1,
                                        maximum=1.0,
                                        step=0.05,
                                        value=0.95,
                                        label="Top P",
                                        info="Controls diversity via nucleus sampling",
                                        elem_id="top-p-slider"
                                    )
                                
                            with gr.Row():
                                with gr.Column():
                                    top_k = gr.Slider(
                                        minimum=1,
                                        maximum=100,
                                        step=1,
                                        value=50,
                                        label="Top K",
                                        info="Limits token selection to top K options",
                                        elem_id="top-k-slider"
                                    )
                                
                            with gr.Row():
                                with gr.Column():
                                    seed = gr.Textbox(
                                        label="Seed",
                                        placeholder="Enter a number for reproducible results (optional)",
                                        elem_id="seed-input"
                                    )

                generate_button_clone.click(
                    fn=voice_clone,
                    inputs=[
                        text_input,
                        prompt_text_input,
                        prompt_wav_upload,
                        prompt_wav_record,
                        temperature,
                        top_p,
                        top_k,
                        seed
                    ],
                    outputs=audio_output_clone
                )

            # Voice Creation Tab
            with gr.TabItem("Voice Creation"):
                gr.Markdown("### Create a custom voice with adjustable parameters")

                # Two-column layout for text input and generation output
                with gr.Row():
                    # Text input in the left column
                    with gr.Column(elem_classes="block"):
                        gr.Markdown("### Text Input")
                        text_input_creation = gr.Textbox(
                            label="Text to Synthesize",
                            lines=5,
                            placeholder="Enter the text you want to convert to speech...",
                            value="Welcome to Spark-TTS. You can generate a customized voice by adjusting parameters such as pitch and speaking rate.",
                            elem_id="creation-text"
                        )
                        
                        # Audio output below text input
                        gr.Markdown("### Output")
                        audio_output_creation = gr.Audio(
                            label="Generated Audio"
                        )
                        
                        create_button = gr.Button("Create Voice", elem_classes="primary")
                    
                    # Controls in the right column - divided into two rows
                    with gr.Column():
                        # Voice characteristics controls
                        with gr.Column(elem_classes="block"):
                            gr.Markdown("### Voice Characteristics")
                            
                            gender = gr.Radio(
                                choices=["male", "female"],
                                value="male",
                                label="Voice Gender",
                                elem_id="gender-select"
                            )
                            
                            with gr.Row():
                                with gr.Column():
                                    pitch = gr.Slider(
                                        minimum=1,
                                        maximum=5,
                                        step=1,
                                        value=3,
                                        label="Pitch Level",
                                        elem_id="pitch-slider"
                                    )
                                    
                                    pitch_labels = gr.HTML(
                                        '<div style="display:flex; justify-content:space-between; margin-top:-15px; font-size:12px;"><span>Very Low</span><span>Low</span><span>Normal</span><span>High</span><span>Very High</span></div>'
                                    )
                                
                            with gr.Row():
                                with gr.Column():
                                    speed = gr.Slider(
                                        minimum=1,
                                        maximum=5,
                                        step=1,
                                        value=3,
                                        label="Speaking Rate",
                                        elem_id="speed-slider"
                                    )
                                    
                                    speed_labels = gr.HTML(
                                        '<div style="display:flex; justify-content:space-between; margin-top:-15px; font-size:12px;"><span>Very Slow</span><span>Slow</span><span>Normal</span><span>Fast</span><span>Very Fast</span></div>'
                                    )
                        
                        # Generation parameters controls
                        with gr.Column(elem_classes="block"):
                            gr.Markdown("### Generation Parameters")
                            
                            with gr.Row():
                                with gr.Column():
                                    temperature_creation = gr.Slider(
                                        minimum=0.1,
                                        maximum=1.2,
                                        step=0.05,
                                        value=0.7,
                                        label="Temperature",
                                        info="Controls randomness (higher = more diverse outputs). Lower values produce more consistent results with less silence.",
                                        elem_id="creation-temperature-slider"
                                    )
                                
                            with gr.Row():
                                with gr.Column():
                                    top_p_creation = gr.Slider(
                                        minimum=0.1,
                                        maximum=1.0,
                                        step=0.05,
                                        value=0.95,
                                        label="Top P",
                                        info="Controls diversity via nucleus sampling",
                                        elem_id="creation-top-p-slider"
                                    )
                                
                            with gr.Row():
                                with gr.Column():
                                    top_k_creation = gr.Slider(
                                        minimum=1,
                                        maximum=100,
                                        step=1,
                                        value=50,
                                        label="Top K",
                                        info="Limits token selection to top K options",
                                        elem_id="creation-top-k-slider"
                                    )
                                
                            with gr.Row():
                                with gr.Column():
                                    seed_creation = gr.Textbox(
                                        label="Seed",
                                        placeholder="Enter a number for reproducible results (optional)",
                                        elem_id="creation-seed-input"
                                    )

                create_button.click(
                    fn=voice_creation,
                    inputs=[
                        text_input_creation,
                        gender,
                        pitch,
                        speed,
                        temperature_creation,
                        top_p_creation,
                        top_k_creation,
                        seed_creation
                    ],
                    outputs=audio_output_creation
                )

            # File Manager Tab
            with gr.TabItem("Manage Files"):
                gr.Markdown(
                    "### Manage Your Generated Audio Files"
                )
                
                with gr.Row():
                    with gr.Column(scale=3):
                        file_list = gr.Dataframe(
                            headers=["Filename", "Date", "Size (KB)"],
                            datatype=["str", "str", "str"],
                            col_count=(3, "fixed"),
                            row_count=(10, "dynamic"),
                            interactive=False,
                            elem_id="file-list-table"
                        )
                    
                    with gr.Column(scale=2):
                        selected_file = gr.State(None)
                        preview_audio = gr.Audio(
                            label="Preview Selected File",
                            interactive=False,
                            elem_id="preview-audio"
                        )
                
                with gr.Row():
                    refresh_btn = gr.Button("Refresh List", variant="secondary")
                    delete_btn = gr.Button("Delete Selected", variant="stop")
                    delete_all_btn = gr.Button("Delete All Files", variant="stop")
                    download_btn = gr.Button("Download Selected", variant="secondary")
                
                # Function to list files
                def list_files():
                    files = []
                    for file_path in glob.glob(os.path.join(TEMP_DIR, "*.wav")):
                        file_info = os.stat(file_path)
                        file_date = datetime.fromtimestamp(file_info.st_mtime).strftime("%Y-%m-%d %H:%M:%S")
                        file_size = f"{file_info.st_size / 1024:.1f}"
                        files.append([os.path.basename(file_path), file_date, file_size])
                    
                    # Sort by date, newest first (assuming date is in column 1)
                    files.sort(key=lambda x: x[1], reverse=True)
                    return files
                
                # Function to get file path from filename
                def get_file_path(filename):
                    if not filename:
                        return None
                    return os.path.join(TEMP_DIR, filename)
                
                # Function to preview file
                def preview_file(evt: gr.SelectData, files):
                    if files is None or files.empty or evt.index[0] >= len(files):
                        return None, None
                    
                    filename = files.iloc[evt.index[0]][0]
                    file_path = get_file_path(filename)
                    
                    if file_path and os.path.exists(file_path):
                        return file_path, filename
                    return None, None
                
                # Function to delete file
                def delete_file(filename):
                    if not filename:
                        return "No file selected", list_files()
                    
                    file_path = get_file_path(filename)
                    if not file_path:
                        return "Invalid file path", list_files()
                    
                    try:
                        if os.path.exists(file_path):
                            os.remove(file_path)
                            return f"Deleted file: {filename}", list_files()
                        else:
                            return "File not found", list_files()
                    except Exception as e:
                        return f"Error deleting file: {str(e)}", list_files()
                
                # Function to delete all files
                def delete_all_files():
                    try:
                        count = 0
                        for file_path in glob.glob(os.path.join(TEMP_DIR, "*.wav")):
                            os.remove(file_path)
                            count += 1
                        return f"Deleted {count} files", list_files()
                    except Exception as e:
                        return f"Error deleting files: {str(e)}", list_files()
                
                # Function to download file
                def download_file(filename):
                    if not filename:
                        return "No file selected"
                    
                    file_path = get_file_path(filename)
                    if not file_path or not os.path.exists(file_path):
                        return "File not found"
                    
                    return file_path
                
                # Event handlers
                file_list.select(preview_file, [file_list], [preview_audio, selected_file])
                refresh_btn.click(list_files, [], [file_list])
                delete_btn.click(delete_file, [selected_file], [gr.Textbox(visible=False), file_list])
                delete_all_btn.click(delete_all_files, [], [gr.Textbox(visible=False), file_list])
                download_btn.click(download_file, [selected_file], [gr.File(visible=True, label="Download")])
                
                # Initialize with existing files
                file_list.value = list_files()

    return demo


def parse_arguments():
    """
    Parse command-line arguments such as model directory and device ID.
    """
    parser = argparse.ArgumentParser(description="Spark TTS Gradio server.")
    parser.add_argument(
        "--model_dir",
        type=str,
        default="pretrained_models/Spark-TTS-0.5B",
        help="Path to the model directory."
    )
    parser.add_argument(
        "--device",
        type=int,
        default=0,
        help="ID of the GPU device to use (e.g., 0 for cuda:0)."
    )
    parser.add_argument(
        "--server_name",
        type=str,
        default="0.0.0.0",
        help="Server host/IP for Gradio app."
    )
    parser.add_argument(
        "--server_port",
        type=int,
        default=7860,
        help="Server port for Gradio app."
    )
    return parser.parse_args()


def kill_process_on_port(port):
    """
    Kill the process running on a specific port (Windows-specific version).
    Returns True if a process was killed, False otherwise.
    """
    try:
        # For Windows, use a more direct approach
        print(f"Checking if port {port} is in use...")
        
        # Run netstat to find the PID
        result = subprocess.run(
            f'netstat -ano | findstr :{port} | findstr LISTENING',
            shell=True,
            capture_output=True,
            text=True
        )
        
        # Check if we found any process
        if result.stdout.strip():
            lines = result.stdout.strip().split('\n')
            for line in lines:
                parts = line.strip().split()
                if len(parts) >= 5:  # Ensure there are enough parts in the line
                    pid = parts[-1]
                    print(f"Found process using port {port}: PID {pid}")
                    
                    # Kill the process
                    kill_result = subprocess.run(
                        f'taskkill /F /PID {pid}',
                        shell=True,
                        capture_output=True,
                        text=True
                    )
                    
                    if kill_result.returncode == 0:
                        print(f"Successfully terminated process {pid} using port {port}")
                        return True
                    else:
                        print(f"Failed to terminate process: {kill_result.stderr}")
        else:
            print(f"No process found using port {port}")
            return False
            
    except Exception as e:
        print(f"Error checking/killing process on port {port}: {e}")
        return False
        
    return False


def get_instance(model_dir="pretrained_models/Spark-TTS-0.5B", device=0):
    """Get or initialize the model instance."""
    global model_instance
    
    if model_instance is None:
        initialize_model(model_dir, device)
    
    return model_instance


if __name__ == "__main__":
    # Parse command-line arguments
    args = parse_arguments()
    
    # Kill any process using the port
    kill_process_on_port(args.server_port)
    
    # Build the Gradio demo by specifying the model directory and GPU device
    demo = build_ui(
        model_dir=args.model_dir,
        device=args.device
    )
    
    # Launch the Gradio interface
    demo.launch(
        server_name=args.server_name,
        server_port=args.server_port,
    )