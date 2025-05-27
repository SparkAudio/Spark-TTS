#!/bin/bash
if [ "$test" == "true" ]; then
  python test.py
else
  python -m cli.inference \
    --text "$text" \
    --device 0 \
    --save_dir "/usr/src/app/shared/results" \
    --model_dir "$model_dir" \
    --prompt_text "$prompt_text" \
    --prompt_speech_path "$prompt_speech_path"
fi
