#!/bin/bash
python -m cli.inference \
  --text "$text" \
  --device 0 \
  --save_dir "/usr/src/app/example/results" \
  --model_dir "$model_dir" \
  --prompt_text "$prompt_text" \
  --prompt_speech_path "$prompt_speech_path"