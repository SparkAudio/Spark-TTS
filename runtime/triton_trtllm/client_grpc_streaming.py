# https://github.com/triton-inference-server/client/tree/main/src/python/examples

import argparse
from functools import partial
import os
import queue
import uuid

import numpy as np
import soundfile as sf

import tritonclient.grpc as grpcclient
from tritonclient.utils import np_to_triton_dtype, InferenceServerException


class UserData:
    def __init__(self):
        self._completed_requests = queue.Queue()


def callback(user_data, result, error):
    if error:
        user_data._completed_requests.put(error)
    else:
        user_data._completed_requests.put(result)


def tts(
    server_url: str = "localhost:8000",
    reference_audio: str = "/prompt_audio.wav",
    reference_text: str = "吃燕窝就选燕之屋，本节目由26年专注高品质燕窝的燕之屋冠名播出。豆奶牛奶换着喝，营养更均衡，本节目由豆本豆豆奶特约播出。",
    target_text: str = "身临其境，换新体验。塑造开源语音合成新范式，让智能语音更自然。",
    model_name: str = "spark_tts_decoupled",
    output_audio: str = "grpc_output.wav",
    verbose: bool = False,
    chunk_overlap_duration: float = 0.1,
):
    cross_fade_samples = int(chunk_overlap_duration * 16000)
    fade_out = np.linspace(1, 0, cross_fade_samples)
    fade_in = np.linspace(0, 1, cross_fade_samples)

    waveform, sr = sf.read(reference_audio)
    duration = sf.info(reference_audio).duration
    assert sr == 16000, "sample rate hardcoded in server"

    samples = np.array(waveform, dtype=np.float32)
    inputs = prepare_grpc_sdk_request(
        samples, reference_text, target_text, duration, sample_rate=sr
    )

    user_data = UserData()

    # https://github.com/triton-inference-server/client/blob/main/src/python/examples/simple_http_aio_infer_client.py
    triton_client = grpcclient.InferenceServerClient(url=server_url, verbose=verbose)
    # Establish stream
    triton_client.start_stream(callback=partial(callback, user_data))
    outputs = [grpcclient.InferRequestedOutput("waveform")]
    req_id = str(uuid.uuid4())
    triton_client.async_stream_infer(
        model_name,
        inputs,
        request_id=req_id,
        outputs=outputs,
        enable_empty_final_response=True,  # u need it !!!
    )

    audios = []
    while True:
        data_item = user_data._completed_requests.get()
        if isinstance(data_item, InferenceServerException):
            raise data_item
        data = data_item.get_response()
        print(f"parameters:{data.parameters}")
        final = data.parameters["triton_final_response"].bool_param
        if final is True:
            break

        request_id = data_item.get_response().id
        assert request_id == req_id, f"request id mismatch {request_id} != {req_id}"
        audio = data_item.as_numpy("waveform").reshape(-1)
        print(audio.shape)
        audios.append(audio)

    for i, audio in enumerate(audios):
        if i == 0:
            new_audio = audio[:-cross_fade_samples]
        else:
            cross_faded_overlap = audio[:cross_fade_samples] * fade_in + audios[i - 1][-cross_fade_samples:] * fade_out
            new_audio = np.concatenate([new_audio, cross_faded_overlap, audio[cross_fade_samples:-cross_fade_samples]])
    new_audio = np.concatenate([new_audio, audio[-cross_fade_samples:]])

    sf.write(output_audio, new_audio, 16000, "PCM_16")
    print(f"save audio to {output_audio}")

    triton_client.close()

def prepare_grpc_sdk_request(
    waveform,
    reference_text,
    target_text,
    duration: float,
    sample_rate=16000,
    padding_duration: int = None,
):
    assert len(waveform.shape) == 1, "waveform should be 1D"
    lengths = np.array([[len(waveform)]], dtype=np.int32)
    if padding_duration:
        # padding to nearset 10 seconds
        samples = np.zeros(
            (
                1,
                padding_duration * sample_rate * ((int(duration) // padding_duration) + 1),
            ),
            dtype=np.float32,
        )

        samples[0, : len(waveform)] = waveform
    else:
        samples = waveform

    samples = samples.reshape(1, -1).astype(np.float32)

    """
    data = {
        "inputs": [
            {
                "name": "reference_wav",
                "shape": samples.shape,
                "datatype": "FP32",
                "data": samples.tolist(),
            },
            {
                "name": "reference_wav_len",
                "shape": lengths.shape,
                "datatype": "INT32",
                "data": lengths.tolist(),
            },
            {
                "name": "reference_text",
                "shape": [1, 1],
                "datatype": "BYTES",
                "data": [reference_text],
            },
            {"name": "target_text", "shape": [1, 1], "datatype": "BYTES", "data": [target_text]},
        ]
    }
    """
    inputs = [
        grpcclient.InferInput("reference_wav", samples.shape, np_to_triton_dtype(samples.dtype)),
        grpcclient.InferInput(
            "reference_wav_len", lengths.shape, np_to_triton_dtype(lengths.dtype)
        ),
        grpcclient.InferInput("reference_text", [1, 1], "BYTES"),
        grpcclient.InferInput("target_text", [1, 1], "BYTES"),
    ]
    inputs[0].set_data_from_numpy(samples)
    inputs[1].set_data_from_numpy(lengths)

    input_data_numpy = np.array([reference_text], dtype=object)
    input_data_numpy = input_data_numpy.reshape((1, 1))
    inputs[2].set_data_from_numpy(input_data_numpy)

    input_data_numpy = np.array([target_text], dtype=object)
    input_data_numpy = input_data_numpy.reshape((1, 1))
    inputs[3].set_data_from_numpy(input_data_numpy)

    return inputs


def get_args():
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument(
        "--server-url",
        type=str,
        default="localhost:8000",
        help="Address of the server",
    )

    parser.add_argument(
        "--reference-audio",
        type=str,
        default="../../example/prompt_audio.wav",
        help="Path to a single audio file. It can't be specified at the same time with --manifest-dir",
    )

    parser.add_argument(
        "--reference-text",
        type=str,
        default="吃燕窝就选燕之屋，本节目由26年专注高品质燕窝的燕之屋冠名播出。豆奶牛奶换着喝，营养更均衡，本节目由豆本豆豆奶特约播出。",
        help="",
    )

    parser.add_argument(
        "--target-text",
        type=str,
        default="身临其境，换新体验。塑造开源语音合成新范式，让智能语音更自然。",
        help="",
    )

    parser.add_argument(
        "--model-name",
        type=str,
        default="spark_tts_decoupled",
        choices=["f5_tts", "spark_tts_decoupled"],
        help="triton model_repo module name to request: transducer for k2, attention_rescoring for wenet offline, streaming_wenet for wenet streaming, infer_pipeline for paraformer large offline",
    )

    parser.add_argument(
        "--output-audio",
        type=str,
        default="output.wav",
        help="Path to save the output audio",
    )

    parser.add_argument(
        "--verbose",
        type=bool,
        default=False,
        help="Path to save the output audio",
    )

    parser.add_argument(
        "--chunk-overlap-duration",
        type=float,
        default=0.1,
        help="Chunk overlap duration",
    )
    
    return parser.parse_args()


"""
python runtime/triton_trtllm/client_grpc_stream.py \
    --server-url r15.modal.host:33585 \
    --reference-audio example/prompt_audio.wav \
    --reference-text "吃燕窝就选燕之屋，本节目由26年专注高品质燕窝的燕之屋冠名播出。豆奶牛奶换着喝，营养更均衡，本节目由豆本豆豆奶特约播出。" \
    --target-text "身临其境，换新体验。塑造开源语音合成新范式，让智能语音更自然。" \
    --model-name spark_tts_decoupled \
    --output-audio output.wav
"""

if __name__ == "__main__":
    args = get_args()
    tts(
        server_url=args.server_url,
        reference_audio=args.reference_audio,
        reference_text=args.reference_text,
        target_text=args.target_text,
        model_name=args.model_name,
        output_audio=args.output_audio,
        verbose=bool(args.verbose),
        chunk_overlap_duration=args.chunk_overlap_duration,
    )
