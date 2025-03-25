# Spark-TTS

<div align="center">
    <a href="https://arxiv.org/pdf/2503.01710"><img src="https://img.shields.io/badge/Paper-ArXiv-red" alt="paper"></a>
    <a href="https://huggingface.co/SparkAudio/Spark-TTS-0.5B"><img src="https://img.shields.io/badge/Hugging%20Face-Model%20Page-yellow" alt="Hugging Face"></a>
    <a href="https://github.com/SparkAudio/Spark-TTS"><img src="https://img.shields.io/badge/Python-3.12+-orange" alt="version"></a>
    <a href="https://github.com/SparkAudio/Spark-TTS"><img src="https://img.shields.io/badge/PyTorch-2.5+-brightgreen" alt="python"></a>
    <a href="https://github.com/SparkAudio/Spark-TTS"><img src="https://img.shields.io/badge/License-Apache%202.0-blue.svg" alt="mit"></a>
</div>

## Overview

Spark-TTS is an advanced text-to-speech system that uses large language models (LLM) for natural-sounding voice synthesis. It is designed to be efficient, flexible, and powerful for both research and production use. The system is based on the paper [Spark-TTS: An Efficient LLM-Based Text-to-Speech Model with Single-Stream Decoupled Speech Tokens](https://arxiv.org/pdf/2503.01710).

This version includes several enhancements to the original model, with added parameter controls, improved UI, and additional features for better usability.

## Key Features

- **Simplicity and Efficiency**: Built on Qwen2.5, Spark-TTS eliminates the need for additional generation models like flow matching. Instead of relying on separate models to generate acoustic features, it directly reconstructs audio from the code predicted by the LLM, streamlining the process and improving efficiency.

- **High-Quality Voice Cloning**: Supports zero-shot voice cloning, which means it can replicate a speaker's voice even without specific training data for that voice. This is ideal for cross-lingual and code-switching scenarios.

- **Bilingual Support**: Supports both Chinese and English, and is capable of zero-shot voice cloning for cross-lingual and code-switching scenarios, enabling the model to synthesize speech in multiple languages with high naturalness and accuracy.

- **Controllable Speech Generation**: Create virtual speakers by adjusting parameters such as gender, pitch, and speaking rate to customize the voice output to your needs.

## Enhancements in this Version

This version of Spark-TTS includes several improvements over the original implementation:

### Advanced Parameter Controls
- **Temperature Controls**: Adjust the randomness of generation with a slider range of 0.1-1.2, with optimized capping to prevent noise/silence generation
- **Top-k and Top-p Sampling**: Precision control over the token selection process, allowing for more deterministic or creative outputs
- **Seed Support**: Added reproducibility through seed values, enabling consistent outputs across multiple runs

### UI and Experience
- **Modern Design**: Clean user interface with proper organization and spacing
- **Detailed UI Labels**: Added helpful explanations for each parameter and slider labels for easier usage
- **Voice Creation Panel**: Enhanced voice customization with visual indicators for pitch and speed levels

### Additional Features
- **File Management System**: Built-in file manager tab allowing you to:
  - Browse generated audio files
  - Preview audio directly in the interface
  - Delete individual or all files
  - Download generated audio with one click
  - Sort files by creation date

- **Audio Processing Improvements**: 
  - Automatic silence trimming from generated audio
  - Organized file naming convention with timestamp and voice type
  - Enhanced error handling and logging

## Installation

### Prerequisites

- Python 3.12+
- PyTorch 2.5+

### Setup

1. Clone the repository:
```sh
git clone https://github.com/h1ddenpr0cess20/Spark-TTS.git
cd Spark-TTS
```

2. Create and activate environment:
```sh
conda create -n sparktts -y python=3.12
conda activate sparktts
pip install -r requirements.txt
```

### Model Download

Download from Hugging Face:
```python
from huggingface_hub import snapshot_download
snapshot_download("SparkAudio/Spark-TTS-0.5B", local_dir="pretrained_models/Spark-TTS-0.5B")
```

Or using git-lfs:
```sh
mkdir -p pretrained_models
git lfs install
git clone https://huggingface.co/SparkAudio/Spark-TTS-0.5B pretrained_models/Spark-TTS-0.5B
```

## Usage

### Basic Command Line Usage

Run the example script:
```sh
cd example
bash infer.sh
```

Or run directly:
```sh
python -m cli.inference \
    --text "text to synthesis." \
    --device 0 \
    --save_dir "path/to/save/audio" \
    --model_dir pretrained_models/Spark-TTS-0.5B \
    --prompt_text "transcript of the prompt audio" \
    --prompt_speech_path "path/to/prompt_audio"
```

### Web UI

Start the web interface with:
```sh
python webui.py --device 0
```

The web UI provides three main tabs:

1. **Voice Clone**: Upload reference audio or record your voice, then enter text to clone the voice
   - Control temperature, top_p, top_k, and seed values
   - Optional reference transcript field for better accuracy

2. **Voice Creation**: Create custom voices with adjustable parameters
   - Choose gender (male/female)
   - Adjust pitch (5 levels from very low to very high)
   - Control speaking rate (5 levels from very slow to very fast)
   - Fine-tune generation with temperature, top_p, top_k, and seed values

3. **Manage Files**: File management system
   - Browse all generated audio files with details (name, date, size)
   - Preview audio directly in the interface
   - Delete individual files or clear all at once
   - Download generated audio files

## Advanced Configuration

The web UI supports additional command-line parameters:
```sh
python webui.py --model_dir "path/to/model" --device 0 --server_name "0.0.0.0" --server_port 7860
```

## Inference Serving

Spark-TTS can be deployed with Nvidia Triton and TensorRT-LLM for production environments. This provides efficient inference with low latency, making it suitable for real-time applications. For detailed instructions, see the runtime documentation.

## Usage Disclaimer

This project provides a zero-shot voice cloning TTS model intended for academic research, educational purposes, and legitimate applications, such as personalized speech synthesis, assistive technologies, and linguistic research.

Please note:
- Do not use this model for unauthorized voice cloning, impersonation, fraud, or any illegal activities.
- Ensure compliance with local laws and regulations when using this model.
- The developers assume no liability for any misuse of this model.

## License

Apache License 2.0