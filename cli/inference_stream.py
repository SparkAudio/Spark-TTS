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
import argparse
import torch
import soundfile as sf
import logging
from datetime import datetime
import platform

from cli.SparkTTS import SparkTTS
from sparktts.utils.audio import merge_numpy_darray


def parse_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description="Run TTS inference.")

    parser.add_argument(
        "--model_dir",
        type=str,
        default="pretrained_models/Spark-TTS-0.5B",
        help="Path to the model directory",
    )
    parser.add_argument(
        "--save_dir",
        type=str,
        default="example/results",
        help="Directory to save generated audio files",
    )
    parser.add_argument("--device", type=int, default=0, help="CUDA device number")
    parser.add_argument("--text", type=str, required=True, help="Text for TTS generation")
    parser.add_argument("--prompt_text", type=str, help="Transcript of prompt audio")
    parser.add_argument(
        "--prompt_speech_path",
        type=str,
        help="Path to the prompt audio file",
    )
    parser.add_argument("--gender", choices=["male", "female"])
    parser.add_argument("--pitch", choices=["very_low", "low", "moderate", "high", "very_high"])
    parser.add_argument("--speed", choices=["very_low", "low", "moderate", "high", "very_high"])
    parser.add_argument(
        "--stream-factor", type=int, default=2, help="Synthesis audios stream factor"
    )
    parser.add_argument(
        "--stream-scale-factor",
        type=float,
        default=1.0,
        help="Synthesis audios stream scale factor",
    )
    parser.add_argument(
        "--max-stream-factor", type=int, default=2, help="Synthesis audios max stream factor"
    )
    parser.add_argument(
        "--token-overlap-len", type=int, default=0, help="Synthesis audios token overlap len"
    )
    return parser.parse_args()


def run_tts(args):
    """Perform TTS inference and save the generated audio."""
    logging.info(f"Using model from: {args.model_dir}")
    logging.info(f"Saving audio to: {args.save_dir}")

    # Ensure the save directory exists
    os.makedirs(args.save_dir, exist_ok=True)

    # Convert device argument to torch.device
    if platform.system() == "Darwin" and torch.backends.mps.is_available():
        # macOS with MPS support (Apple Silicon)
        device = torch.device(f"mps:{args.device}")
        logging.info(f"Using MPS device: {device}")
    elif torch.cuda.is_available():
        # System with CUDA support
        device = torch.device(f"cuda:{args.device}")
        logging.info(f"Using CUDA device: {device}")
    else:
        # Fall back to CPU
        device = torch.device("cpu")
        logging.info("GPU acceleration not available, using CPU")

    # Initialize the model
    model = SparkTTS(
        args.model_dir,
        device,
        stream=True,
        stream_factor=args.stream_factor,
        stream_scale_factor=args.stream_scale_factor,
        max_stream_factor=args.max_stream_factor,
        token_overlap_len=args.token_overlap_len,
    )

    # Generate unique filename using timestamp
    timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
    save_path = os.path.join(args.save_dir, f"{timestamp}.wav")

    logging.info("Starting stream inference...")

    sub_tts_speechs = []
    # Perform inference and save the output audio
    with torch.no_grad():
        batch_stream = model.inference_stream(
            args.text,
            args.prompt_speech_path,
            prompt_text=args.prompt_text,
            gender=args.gender,
            pitch=args.pitch,
            speed=args.speed,
        )
        for item in batch_stream:
            sub_tts_speechs.append(item["tts_speech"])

    output_audio = merge_numpy_darray(sub_tts_speechs)  # [[T],...] -> [T]
    sf.write(save_path, output_audio, samplerate=16000)
    logging.info(f"Audio saved at: {save_path}")


"""
# Inference Overview of Controlled Generation
PYTHONPATH=./ python cli/inference_stream.py \
    --text "身临其境，换新体验。塑造开源语音合成新范式，让智能语音更自然。" \
    --save_dir "example/results" \
    --model_dir ../../models/SparkAudio/Spark-TTS-0.5B \
    --gender female --pitch  moderate --speed high

PYTHONPATH=./ python cli/inference_stream.py \
    --text "万物之始,大道至简,衍化至繁。君不见黄河之水天上来，奔流到海不复回。君不见高堂明镜悲白发，朝如青丝暮成雪。人生得意须尽欢，莫使金樽空对月。天生我材必有用，千金散尽还复来。" \
    --save_dir "example/results" \
    --model_dir ../../models/SparkAudio/Spark-TTS-0.5B \
    --gender female --pitch  moderate --speed high

# Inference Overview of Voice Cloning
# default use static batch is ok
PYTHONPATH=./ python cli/inference_stream.py \
    --text "身临其境，换新体验。塑造开源语音合成新范式，让智能语音更自然。" \
    --save_dir "example/results" \
    --model_dir ../../models/SparkAudio/Spark-TTS-0.5B \
    --prompt_text "吃燕窝就选燕之屋，本节目由26年专注高品质燕窝的燕之屋冠名播出。豆奶牛奶换着喝，营养更均衡，本节目由豆本豆豆奶特约播出。" \
    --prompt_speech_path "example/prompt_audio.wav"

PYTHONPATH=./ python cli/inference_stream.py \
    --text "万物之始,大道至简,衍化至繁。君不见黄河之水天上来，奔流到海不复回。君不见高堂明镜悲白发，朝如青丝暮成雪。人生得意须尽欢，莫使金樽空对月。天生我材必有用，千金散尽还复来。" \
    --save_dir "example/results" \
    --model_dir ../../models/SparkAudio/Spark-TTS-0.5B \
    --prompt_text "吃燕窝就选燕之屋，本节目由26年专注高品质燕窝的燕之屋冠名播出。豆奶牛奶换着喝，营养更均衡，本节目由豆本豆豆奶特约播出。" \
    --prompt_speech_path "example/prompt_audio.wav"
"""
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

    args = parse_args()
    run_tts(args)
