## Nvidia Triton Inference Serving Best Practice for Spark TTS

### Quick Start
Directly launch the service using docker compose.
```sh
docker compose up
```

### Build Image
Build the docker image from scratch. 
```sh
docker build . -f Dockerfile.server -t soar97/triton-spark-tts:25.02
```

### Create Docker Container
```sh
your_mount_dir=/mnt:/mnt
docker run -it --name "spark-tts-server" --gpus all --net host -v $your_mount_dir --shm-size=2g soar97/triton-spark-tts:25.02
```

### Export Models to TensorRT-LLM and Launch Server
Inside docker container, we would follow the official guide of TensorRT-LLM to build TensorRT-LLM engines. See [here](https://github.com/NVIDIA/TensorRT-LLM/tree/main/examples/qwen).

```sh
bash run.sh 0 3
```
### Simple HTTP client
```sh
python3 client_http.py
```

### Benchmark using Dataset
```sh
num_task=2
python3 client_grpc.py --num-tasks $num_task --huggingface-dataset yuekai/seed_tts --split-name wenetspeech4tts
```

### Benchmark Results
Decoding on a single L20 GPU, using 26 different prompt_audio/target_text pairs, total audio duration 169 secs.

| Model | Note   | Concurrency | Avg Latency     | RTF | 
|-------|-----------|-----------------------|---------|--|
| Spark-TTS-0.5B | [Code Commit](https://github.com/SparkAudio/Spark-TTS/tree/4d769ff782a868524f29e0be851ca64f8b22ebf1/runtime/triton_trtllm) | 1                   | 876.24 ms | 0.1362|
| Spark-TTS-0.5B | [Code Commit](https://github.com/SparkAudio/Spark-TTS/tree/4d769ff782a868524f29e0be851ca64f8b22ebf1/runtime/triton_trtllm) | 2                   | 920.97 ms | 0.0737|
| Spark-TTS-0.5B | [Code Commit](https://github.com/SparkAudio/Spark-TTS/tree/4d769ff782a868524f29e0be851ca64f8b22ebf1/runtime/triton_trtllm) | 4                   | 1611.51 ms | 0.0704|