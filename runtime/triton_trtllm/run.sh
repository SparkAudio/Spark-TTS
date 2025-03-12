export PYTHONPATH=../../../Spark-TTS/
export CUDA_VISIBLE_DEVICES=0
stage=$1
stop_stage=$2
echo "Start stage: $stage, Stop stage: $stop_stage"

huggingface_model_local_dir=../../pretrained_models/Spark-TTS-0.5B
trt_dtype=bfloat16
trt_weights_dir=./tllm_checkpoint_${trt_dtype}
trt_engines_dir=./trt_engines_${trt_dtype}

model_repo=./model_repo_test

if [ $stage -le 0 ] && [ $stop_stage -ge 0 ]; then
    echo "Downloading Spark-TTS-0.5B from HuggingFace"
    huggingface-cli download SparkAudio/Spark-TTS-0.5B --local-dir $huggingface_model_local_dir || exit 1
    # pip install -r /workspace_yuekai/spark-tts/Spark-TTS/requirements.txt
fi


if [ $stage -le 1 ] && [ $stop_stage -ge 1 ]; then
    echo "Converting checkpoint to TensorRT weights"
    python scripts/convert_checkpoint.py --model_dir $huggingface_model_local_dir/LLM \
                                --output_dir $trt_weights_dir \
                                --dtype $trt_dtype || exit 1

    echo "Building TensorRT engines"
    trtllm-build --checkpoint_dir $trt_weights_dir \
                --output_dir $trt_engines_dir \
                --max_batch_size 16 \
                --max_num_tokens 32768 \
                --gemm_plugin $trt_dtype || exit 1
fi

if [ $stage -le 2 ] && [ $stop_stage -ge 2 ]; then
    echo "Creating model repository"
    rm -rf $model_repo
    cp -r ./model_repo $model_repo

    ENGINE_PATH=$trt_engines_dir
    MAX_QUEUE_DELAY_MICROSECONDS=0
    MODEL_DIR=$huggingface_model_local_dir
    LLM_TOKENIZER_DIR=$huggingface_model_local_dir/LLM
    BLS_INSTANCE_NUM=4
    TRITON_MAX_BATCH_SIZE=16

    python3 scripts/fill_template.py -i ${model_repo}/vocoder/config.pbtxt model_dir:${MODEL_DIR},triton_max_batch_size:${TRITON_MAX_BATCH_SIZE},max_queue_delay_microseconds:${MAX_QUEUE_DELAY_MICROSECONDS}
    python3 scripts/fill_template.py -i ${model_repo}/audio_tokenizer/config.pbtxt model_dir:${MODEL_DIR},triton_max_batch_size:${TRITON_MAX_BATCH_SIZE},max_queue_delay_microseconds:${MAX_QUEUE_DELAY_MICROSECONDS}
    python3 scripts/fill_template.py -i ${model_repo}/spark_tts/config.pbtxt bls_instance_num:${BLS_INSTANCE_NUM},llm_tokenizer_dir:${LLM_TOKENIZER_DIR},triton_max_batch_size:${TRITON_MAX_BATCH_SIZE},max_queue_delay_microseconds:${MAX_QUEUE_DELAY_MICROSECONDS}
    python3 scripts/fill_template.py -i ${model_repo}/tensorrt_llm/config.pbtxt triton_backend:tensorrtllm,triton_max_batch_size:${TRITON_MAX_BATCH_SIZE},decoupled_mode:False,max_beam_width:1,engine_dir:${ENGINE_PATH},max_tokens_in_paged_kv_cache:2560,max_attention_window_size:2560,kv_cache_free_gpu_mem_fraction:0.5,exclude_input_in_output:True,enable_kv_cache_reuse:False,batching_strategy:inflight_fused_batching,max_queue_delay_microseconds:${MAX_QUEUE_DELAY_MICROSECONDS},encoder_input_features_data_type:TYPE_FP16,logits_datatype:TYPE_FP32

fi

if [ $stage -le 3 ] && [ $stop_stage -ge 3 ]; then
    echo "Starting Triton server"
    tritonserver --model-repository ${model_repo}
fi


if [ $stage -le 4 ] && [ $stop_stage -ge 4 ]; then
    echo "Running client"
    num_task=2
    python3 client_grpc.py \
        --server-addr localhost \
        --model-name spark_tts \
        --num-tasks $num_task \
        --log-dir ./log_concurrent_tasks_${num_task}
fi