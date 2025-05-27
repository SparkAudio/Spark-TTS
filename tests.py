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
                    "--text", "Where does sound end and silence begin? Perhaps that is where the most important thoughts are born. Listen carefully - each word carries more than just meaning. This is just the beginning of a story you don't know yet.",
                    "--prompt_text", "Ladies and gentlemen, I am delighted to reet members and guests of the general assembly of the international exhibitions bureau. Russia has a long and rich experience of participation in the world expo movement.",
                    "--device", "0",
                    "--save_dir", work_dir,
                    "--model_dir", "pretrained_models/Spark-TTS-0.5B",
                    "--prompt_speech_path", f"{work_dir}/prompt_audio.wav"])
    
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
