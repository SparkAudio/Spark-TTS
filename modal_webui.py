"""Deploy the Gradio web UI to Modal.

This script mirrors the approach in Modal's Streamlit example. It packages
``webui.py`` inside a container image and exposes it via ``modal serve`` or
``modal deploy``.
"""

import os
import shlex
import subprocess
from pathlib import Path

from huggingface_hub import snapshot_download
import modal


# Path to the local Gradio interface script
webui_script_local_path = Path(__file__).parent / "webui.py"
# Location inside the container
webui_script_remote_path = "/root/webui.py"

# Build the container image with dependencies and the web UI script
image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("ffmpeg", "git-lfs")
    .run_commands("git lfs install")
    .pip_install_from_requirements("requirements.txt")
    .add_local_file(webui_script_local_path, webui_script_remote_path)
)

app = modal.App(name="spark-tts-webui", image=image)

# Ensure the webui script is present
if not webui_script_local_path.exists():
    raise RuntimeError(
        "webui.py not found! Place modal_webui.py next to webui.py."
    )


@app.function(gpu="A100")
@modal.web_server(7860)
@modal.concurrent(max_inputs=10) # Added this line
def run(model_dir="pretrained_models/Spark-TTS-0.5B", device: int = 0):
    """Start the Gradio UI inside a Modal container."""

    # Ensure the model is downloaded
    model_path = Path(model_dir)
    if not model_path.exists():
        os.makedirs(model_dir, exist_ok=True)
        snapshot_download(
            "SparkAudio/Spark-TTS-0.5B",
            local_dir=model_dir,
        )

    target = shlex.quote(webui_script_remote_path)
    model_dir = shlex.quote(str(model_path))
    cmd = (
        f"python {target} --model_dir {model_dir} --device {device} "
        f"--server_name 0.0.0.0 --server_port 7860"
    )
    subprocess.Popen(cmd, shell=True)

