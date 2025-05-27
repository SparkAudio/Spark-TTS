import logging
import subprocess
import os
import numpy as np
from sparktts.utils.audio import load_audio

def test1():
    app_dir = "/usr/src/app"
    work_dir = f"{app_dir}/tests/test1"
    
    logging.info("Generate file")
    subprocess.run(["python", "-m", "cli.inference",
                    "--text", "身临其境，换新体验。塑造开源语音合成新范式，让智能语音更自然。",
                    "--prompt_text", "吃燕窝就选燕之屋，本节目由26年专注高品质燕窝的燕之屋冠名播出。豆奶牛奶换着喝，营养更均衡，本节目由豆本豆豆奶特约播出。",
                    "--device", "0",
                    "--save_dir", work_dir,
                    "--model_dir", "pretrained_models/Spark-TTS-0.5B",
                    "--prompt_speech_path", f"{app_dir}/example/prompt_audio.wav"])
    
    with open("last_file.txt", 'r') as file:
        save_path = os.path.join(work_dir, file.read())
    test_audio = load_audio(save_path)
    reference_audio = load_audio(f"{work_dir}/reference_audio.wav")

    logging.info("Comparison")
    reference_len = len(reference_audio)
    test_len = len(test_audio)
    if test_len != reference_len:
        logging.warning("Different length of audio files!")
        if test_len < reference_len:
            size = test_len
        else:
            size = reference_len
    else:
        size = reference_len

    test_audio = test_audio[:size]
    reference_audio = reference_audio[:size]
    max_error = np.max(np.abs(test_audio - reference_audio))
    
    logging.info(f"The maximum error amounted to {max_error}")

    return max_error

def test_all():
    test1()
