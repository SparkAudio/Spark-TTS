export PYTHONPATH=../../../Spark-TTS/
export CUDA_VISIBLE_DEVICES=0
stage=$1
stop_stage=$2
service_type=$3
echo "Start stage: $stage, Stop stage: $stop_stage service_type: $service_type"

huggingface_model_local_dir=../../pretrained_models/Spark-TTS-0.5B
trt_dtype=bfloat16
trt_weights_dir=./tllm_checkpoint_${trt_dtype}
trt_engines_dir=./trt_engines_${trt_dtype}

model_repo=./model_repo_test

if [ $stage -le 0 ] && [ $stop_stage -ge 0 ]; then
    echo "Downloading Spark-TTS-0.5B from HuggingFace"
    huggingface-cli download SparkAudio/Spark-TTS-0.5B --local-dir $huggingface_model_local_dir || exit 1
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
    mkdir -p $model_repo
    spark_tts_dir="spark_tts"
    if [ "$service_type" == "streaming" ]; then
        spark_tts_dir="spark_tts_decoupled"
    fi 
    cp -r ./model_repo/${spark_tts_dir} $model_repo
    cp -r ./model_repo/audio_tokenizer $model_repo
    cp -r ./model_repo/tensorrt_llm $model_repo
    cp -r ./model_repo/vocoder $model_repo

    ENGINE_PATH=$trt_engines_dir
    MAX_QUEUE_DELAY_MICROSECONDS=0
    MODEL_DIR=$huggingface_model_local_dir
    LLM_TOKENIZER_DIR=$huggingface_model_local_dir/LLM
    BLS_INSTANCE_NUM=4
    TRITON_MAX_BATCH_SIZE=16
    DECOUPLED_MODE=False
    python3 scripts/fill_template.py -i ${model_repo}/vocoder/config.pbtxt model_dir:${MODEL_DIR},triton_max_batch_size:${TRITON_MAX_BATCH_SIZE},max_queue_delay_microseconds:${MAX_QUEUE_DELAY_MICROSECONDS}
    python3 scripts/fill_template.py -i ${model_repo}/audio_tokenizer/config.pbtxt model_dir:${MODEL_DIR},triton_max_batch_size:${TRITON_MAX_BATCH_SIZE},max_queue_delay_microseconds:${MAX_QUEUE_DELAY_MICROSECONDS}
    if [ "$service_type" == "streaming" ]; then
        DECOUPLED_MODE=True
        AUDIO_CHUNK_DURATION=1.0
        MAX_AUDIO_CHUNK_DURATION=8.0
        AUDIO_CHUNK_SIZE_SCALE_FACTOR=2.0
        AUDIO_CHUNK_OVERLAP_DURATION=0.1
        python3 scripts/fill_template.py -i ${model_repo}/${spark_tts_dir}/config.pbtxt bls_instance_num:${BLS_INSTANCE_NUM},llm_tokenizer_dir:${LLM_TOKENIZER_DIR},triton_max_batch_size:${TRITON_MAX_BATCH_SIZE},max_queue_delay_microseconds:${MAX_QUEUE_DELAY_MICROSECONDS},audio_chunk_duration:${AUDIO_CHUNK_DURATION},max_audio_chunk_duration:${MAX_AUDIO_CHUNK_DURATION},audio_chunk_size_scale_factor:${AUDIO_CHUNK_SIZE_SCALE_FACTOR},audio_chunk_overlap_duration:${AUDIO_CHUNK_OVERLAP_DURATION}
    else
        python3 scripts/fill_template.py -i ${model_repo}/${spark_tts_dir}/config.pbtxt bls_instance_num:${BLS_INSTANCE_NUM},llm_tokenizer_dir:${LLM_TOKENIZER_DIR},triton_max_batch_size:${TRITON_MAX_BATCH_SIZE},max_queue_delay_microseconds:${MAX_QUEUE_DELAY_MICROSECONDS}
    fi
    python3 scripts/fill_template.py -i ${model_repo}/tensorrt_llm/config.pbtxt triton_backend:tensorrtllm,triton_max_batch_size:${TRITON_MAX_BATCH_SIZE},decoupled_mode:${DECOUPLED_MODE},max_beam_width:1,engine_dir:${ENGINE_PATH},max_tokens_in_paged_kv_cache:2560,max_attention_window_size:2560,kv_cache_free_gpu_mem_fraction:0.5,exclude_input_in_output:True,enable_kv_cache_reuse:False,batching_strategy:inflight_fused_batching,max_queue_delay_microseconds:${MAX_QUEUE_DELAY_MICROSECONDS},encoder_input_features_data_type:TYPE_FP16,logits_datatype:TYPE_FP32

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


if [ $stage -le 5 ] && [ $stop_stage -ge 5 ]; then
    echo "Running streaming grpc client"
    python client_grpc_streaming.py \
        --server-url localhost:8001 \
        --reference-audio ../../example/prompt_audio.wav \
        --reference-text "吃燕窝就选燕之屋，本节目由26年专注高品质燕窝的燕之屋冠名播出。豆奶牛奶换着喝，营养更均衡，本节目由豆本豆豆奶特约播出。" \
        --target-text "身临其境，换新体验。塑造开源语音合成新范式，让智能语音更自然。" \
        --model-name spark_tts_decoupled \
        --chunk-overlap-duration 0.1 \
        --output-audio output.wav
fi