# Copyright (c) 2025 SparkAudio
#               2025 Xinsheng Wang (w.xinshawn@gmail.com)
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import logging
import math
import re
from threading import Thread
import uuid
import torch
from typing import Generator, Tuple
from pathlib import Path
from transformers import AutoTokenizer, AutoModelForCausalLM

from sparktts.utils import ThreadSafeDict
from sparktts.utils.file import load_config
from sparktts.models.audio_tokenizer import BiCodecTokenizer
from sparktts.utils.token_parser import LEVELS_MAP, GENDER_MAP, TASK_TOKEN_MAP
from cli.streamer import TokenStreamer


class SparkTTS:
    """
    Spark-TTS for text-to-speech generation.
    """

    def __init__(
        self,
        model_dir: Path,
        device: torch.device = torch.device("cuda:0"),
        stream: bool = False,
        stream_factor: int = 2,
        stream_scale_factor: float = 1.0,
        max_stream_factor: int = 2,
        token_overlap_len: int = 0,
        input_frame_rate: int = 25,
        **kwargs,
    ):
        """
        Initializes the SparkTTS model with the provided configurations and device.

        Args:
            model_dir (Path): Directory containing the model and config files.
            device (torch.device): The device (CPU/GPU) to run the model on.
        """
        if stream is True:
            # fast path to check params
            # rtf and decoding related
            assert (
                stream_factor >= 2
            ), f"stream_factor must >=2 increase for better speech quality, but rtf slow (speech quality vs rtf)"
            self.stream_factor = stream_factor
            self.max_stream_factor = max_stream_factor
            assert (
                stream_scale_factor >= 1.0
            ), "stream_scale_factor should be greater than 1, change it according to your actual rtf"
            self.stream_scale_factor = stream_scale_factor  # scale speed
            assert (
                token_overlap_len >= 0
            ), "token_overlap_len should be greater than 0, change it according to your actual rtf"
            self.token_overlap_len = token_overlap_len
            self.input_frame_rate = input_frame_rate

        self.device = device
        self.model_dir = model_dir
        self.configs = load_config(f"{model_dir}/config.yaml")
        self.sample_rate = self.configs["sample_rate"]
        self._initialize_inference()
        self.start_global_token_id = self.tokenizer.encode("<|start_global_token|>")[0]
        self.start_semantic_token_id = self.tokenizer.encode("<|start_semantic_token|>")[0]
        logging.debug(f"start_global_token_id:{self.start_global_token_id} start_semantic_token_id:{self.start_semantic_token_id}")

    def _initialize_inference(self):
        """Initializes the tokenizer, model, and audio tokenizer for inference."""
        self.tokenizer = AutoTokenizer.from_pretrained(f"{self.model_dir}/LLM")
        self.model = AutoModelForCausalLM.from_pretrained(f"{self.model_dir}/LLM")
        self.audio_tokenizer = BiCodecTokenizer(self.model_dir, device=self.device)
        self.model.to(self.device)

    def process_prompt(
        self,
        text: str,
        prompt_speech_path: Path,
        prompt_text: str = None,
    ) -> Tuple[str, torch.Tensor]:
        """
        Process input for voice cloning.

        Args:
            text (str): The text input to be converted to speech.
            prompt_speech_path (Path): Path to the audio file used as a prompt.
            prompt_text (str, optional): Transcript of the prompt audio.

        Return:
            Tuple[str, torch.Tensor]: Input prompt; global tokens
        """

        global_token_ids, semantic_token_ids = self.audio_tokenizer.tokenize(prompt_speech_path)
        global_tokens = "".join([f"<|bicodec_global_{i}|>" for i in global_token_ids.squeeze()])

        # Prepare the input tokens for the model
        if prompt_text is not None:
            semantic_tokens = "".join(
                [f"<|bicodec_semantic_{i}|>" for i in semantic_token_ids.squeeze()]
            )
            inputs = [
                TASK_TOKEN_MAP["tts"],
                "<|start_content|>",
                prompt_text,
                text,
                "<|end_content|>",
                "<|start_global_token|>",
                global_tokens,
                "<|end_global_token|>",
                "<|start_semantic_token|>",
                semantic_tokens,
            ]
        else:
            inputs = [
                TASK_TOKEN_MAP["tts"],
                "<|start_content|>",
                text,
                "<|end_content|>",
                "<|start_global_token|>",
                global_tokens,
                "<|end_global_token|>",
            ]

        inputs = "".join(inputs)

        return inputs, global_token_ids

    def process_prompt_control(
        self,
        gender: str,
        pitch: str,
        speed: str,
        text: str,
    ):
        """
        Process input for voice creation.

        Args:
            gender (str): female | male.
            pitch (str): very_low | low | moderate | high | very_high
            speed (str): very_low | low | moderate | high | very_high
            text (str): The text input to be converted to speech.

        Return:
            str: Input prompt
        """
        assert gender in GENDER_MAP.keys()
        assert pitch in LEVELS_MAP.keys()
        assert speed in LEVELS_MAP.keys()

        gender_id = GENDER_MAP[gender]
        pitch_level_id = LEVELS_MAP[pitch]
        speed_level_id = LEVELS_MAP[speed]

        pitch_label_tokens = f"<|pitch_label_{pitch_level_id}|>"
        speed_label_tokens = f"<|speed_label_{speed_level_id}|>"
        gender_tokens = f"<|gender_{gender_id}|>"

        attribte_tokens = "".join([gender_tokens, pitch_label_tokens, speed_label_tokens])

        control_tts_inputs = [
            TASK_TOKEN_MAP["controllable_tts"],
            "<|start_content|>",
            text,
            "<|end_content|>",
            "<|start_style_label|>",
            attribte_tokens,
            "<|end_style_label|>",
        ]

        return "".join(control_tts_inputs)

    def token2wav(self, generated_ids: torch.Tensor, gender: str, global_token_ids: torch.Tensor):
        """
        generated_ids -- tokenizer.decode --> sematic tokens + global tokens  -- audio_tokenizer.detokenize --> waveform
        """
        #print("generated_ids", generated_ids)
        # Decode the generated tokens into text (just a mapping, so quick,don't worry)
        predicts = self.tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
        #print("predicts", predicts)

        # Extract semantic token IDs from the generated text
        pred_semantic_ids = (
            torch.tensor([int(token) for token in re.findall(r"bicodec_semantic_(\d+)", predicts)])
            .long()
            .unsqueeze(0)
        )

        if gender is not None:
            # Tips: generated_id - global_vq_index = 151665
            global_token_ids = (
                torch.tensor(
                    [int(token) for token in re.findall(r"bicodec_global_(\d+)", predicts)]
                )
                .long()
                .unsqueeze(0)
                .unsqueeze(0)
            )

        # Convert semantic tokens back to waveform
        wav = self.audio_tokenizer.detokenize(
            global_token_ids.to(self.device).squeeze(0),
            pred_semantic_ids.to(self.device),
        )

        return wav

    @torch.no_grad()
    def inference(
        self,
        text: str,
        prompt_speech_path: Path = None,
        prompt_text: str = None,
        gender: str = None,
        pitch: str = None,
        speed: str = None,
        temperature: float = 0.8,
        top_k: float = 50,
        top_p: float = 0.95,
    ) -> torch.Tensor:
        """
        Performs inference to generate speech from text, incorporating prompt audio and/or text.

        Args:
            text (str): The text input to be converted to speech.
            prompt_speech_path (Path): Path to the audio file used as a prompt.
            prompt_text (str, optional): Transcript of the prompt audio.
            gender (str): female | male.
            pitch (str): very_low | low | moderate | high | very_high
            speed (str): very_low | low | moderate | high | very_high
            temperature (float, optional): Sampling temperature for controlling randomness. Default is 0.8.
            top_k (float, optional): Top-k sampling parameter. Default is 50.
            top_p (float, optional): Top-p (nucleus) sampling parameter. Default is 0.95.

        Returns:
            torch.Tensor: Generated waveform as a tensor.
        """
        global_token_ids = None
        if gender is not None:
            prompt = self.process_prompt_control(gender, pitch, speed, text)
        else:
            prompt, global_token_ids = self.process_prompt(text, prompt_speech_path, prompt_text)
        model_inputs = self.tokenizer([prompt], return_tensors="pt").to(self.device)

        # Generate speech using the model
        generated_ids = self.model.generate(
            **model_inputs,
            max_new_tokens=3000,
            do_sample=True,
            top_k=top_k,
            top_p=top_p,
            temperature=temperature,
        )

        # Trim the output tokens to remove the input tokens
        generated_ids = [
            output_ids[len(input_ids) :]
            for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
        ]

        wav = self.token2wav(generated_ids, gender, global_token_ids)

        return wav

    @torch.no_grad()
    def inference_stream(
        self,
        text: str,
        prompt_speech_path: Path = None,
        prompt_text: str = None,
        gender: str = None,
        pitch: str = None,
        speed: str = None,
        temperature: float = 0.8,
        top_k: float = 50,
        top_p: float = 0.95,
    ) -> Generator[torch.Tensor, None, None]:
        """
        Performs inference to generate speech from text, incorporating prompt audio and/or text.

        Args:
            text (str): The text input to be converted to speech.
            prompt_speech_path (Path): Path to the audio file used as a prompt.
            prompt_text (str, optional): Transcript of the prompt audio.
            gender (str): female | male.
            pitch (str): very_low | low | moderate | high | very_high
            speed (str): very_low | low | moderate | high | very_high
            temperature (float, optional): Sampling temperature for controlling randomness. Default is 0.8.
            top_k (float, optional): Top-k sampling parameter. Default is 50.
            top_p (float, optional): Top-p (nucleus) sampling parameter. Default is 0.95.


        Returns:
            torch.Tensor: Generated waveform as a tensor generator.
        """
        global_token_ids = None
        if gender is not None:
            prompt = self.process_prompt_control(gender, pitch, speed, text)

        else:
            prompt, global_token_ids = self.process_prompt(text, prompt_speech_path, prompt_text)
        model_inputs = self.tokenizer([prompt], return_tensors="pt").to(self.device)

        # session streamer, skip input prompt
        streamer = TokenStreamer(skip_prompt=True)

        generation_kwargs = dict(
            **model_inputs,
            streamer=streamer,
            max_new_tokens=3000,
            do_sample=True,
            top_k=top_k,
            top_p=top_p,
            temperature=temperature,
        )
        # print("generation_kwargs", generation_kwargs)

        thread = Thread(target=self.model.generate, kwargs=generation_kwargs)
        thread.start()

        is_meet_start_global_token = False
        is_meet_start_semantic_token = False
        controll_gen_global_token_ids = []
        semantic_token_ids = []

        max_batch_size = math.ceil(self.max_stream_factor * self.input_frame_rate)
        batch_size = math.ceil(self.stream_factor * self.input_frame_rate)
        logging.info(f"init batch_size: {batch_size} max_batch_size: {max_batch_size}")

        for token_id in streamer:
            if gender is not None:  # Inference Overview of Controlled Generation
                if is_meet_start_global_token is False and token_id != self.start_global_token_id:
                    continue
                if is_meet_start_global_token is False and token_id == self.start_global_token_id:
                    is_meet_start_global_token = True
                    controll_gen_global_token_ids.append(token_id)
                    continue
                # append global token until meet start_global_token
                if (
                    is_meet_start_global_token is True
                    and is_meet_start_semantic_token is False
                    and token_id != self.start_global_token_id
                ):
                    controll_gen_global_token_ids.append(token_id)

                if is_meet_start_semantic_token is False and token_id != self.start_semantic_token_id:
                    continue
                if is_meet_start_semantic_token is False and token_id == self.start_semantic_token_id:
                    is_meet_start_semantic_token = True
                    continue
                # do batch stream until meet start_semantic_token
                if is_meet_start_semantic_token is True and token_id != self.start_semantic_token_id:
                    # print(controll_gen_global_token_ids)
                    pass

            semantic_token_ids.append(token_id)
            # if len(semantic_token_ids) % batch_size == 0:
            if len(semantic_token_ids) >= batch_size + self.token_overlap_len:
                batch = semantic_token_ids[: batch_size + self.token_overlap_len]
                # Process each batch
                sub_tts_speech = self.token2wav(
                    [controll_gen_global_token_ids + batch], gender, global_token_ids
                )  # one batch
                yield {"tts_speech": sub_tts_speech, "sample_rate": self.sample_rate}
                semantic_token_ids = semantic_token_ids[batch_size:]
                # increase token_hop_len for better speech quality
                batch_size = min(max_batch_size, int(batch_size * self.stream_scale_factor))
                logging.info(
                    f"increase batch_size: {batch_size} token_overlap_len:{self.token_overlap_len}"
                )

        if len(semantic_token_ids) > 0:  # end to finalize
            # Process each batch
            sub_tts_speech = self.token2wav(
                [controll_gen_global_token_ids + semantic_token_ids], gender, global_token_ids
            )  # one batch
            yield {"tts_speech": sub_tts_speech, "sample_rate": self.sample_rate}
            logging.info(f"last batch len: {len(semantic_token_ids)}")

        torch.cuda.empty_cache()
