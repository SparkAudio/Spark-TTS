import logging
import os
import subprocess
from tests import test_all

def run():
    subprocess.run(["python", "-m", "cli.inference",
                    "--text", os.getenv("text"),
                    "--prompt_text", os.getenv("prompt_text"),
                    "--device", "0",
                    "--save_dir", "/usr/src/app/shared/results",
                    "--model_dir", os.getenv("model_dir"),
                    "--prompt_speech_path", os.getenv("prompt_speech_path")])


if __name__ == '__main__':
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
    )

    if os.getenv("test") == "true":
        test_all()
    else:
        run()
