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


import torch
import numpy as np

from pathlib import Path
from typing import Any, Dict, Literal, Optional, Tuple, List, Union
from transformers import Wav2Vec2FeatureExtractor, Wav2Vec2Model

from sparktts.utils.file import load_config
from sparktts.utils.audio import load_audio
from sparktts.models.bicodec import BiCodec


class BiCodecTokenizer:
    """BiCodec tokenizer for handling audio input and tokenization."""

    def __init__(self, model_dir: Path, device: torch.device = None, **kwargs):
        super().__init__()
        """
        Args:
            model_dir: Path to the model directory.
            device: Device to run the model on (default is GPU if available).
        """
        self.device = device
        self.model_dir = model_dir
        self.config = load_config(f"{model_dir}/config.yaml")
        self._initialize_model(**kwargs)

    def _initialize_model(
        self,
        attn_implementation: Optional[Literal["sdpa", "flash_attention_2", "eager"]] = None,
    ):
        """Load and initialize the BiCodec model and Wav2Vec2 feature extractor."""
        self.model = BiCodec.load_from_checkpoint(f"{self.model_dir}/BiCodec").to(self.device)
        self.processor = Wav2Vec2FeatureExtractor.from_pretrained(
            f"{self.model_dir}/wav2vec2-large-xlsr-53"
        )
        self.feature_extractor = (
            Wav2Vec2Model.from_pretrained(
                f"{self.model_dir}/wav2vec2-large-xlsr-53",
                attn_implementation=attn_implementation,
            )
            .to(self.device)
            .eval()
        )
        self.feature_extractor.config.output_hidden_states = True

    def get_ref_clip(self, wav: np.ndarray) -> np.ndarray:
        """Get reference audio clip for speaker embedding."""
        ref_segment_length = (
            int(self.config["sample_rate"] * self.config["ref_segment_duration"])
            // self.config["latent_hop_length"]
            * self.config["latent_hop_length"]
        )
        wav_length = len(wav)

        if ref_segment_length > wav_length:
            # Repeat and truncate to handle insufficient length
            wav = np.tile(wav, ref_segment_length // wav_length + 1)

        return wav[:ref_segment_length]

    def process_audio(self, wav_path: Path) -> Tuple[np.ndarray, np.ndarray]:
        """
        load auido and get reference audio from wav path
        return (wav, wav_ref) # (shape:(seq_len), shape:(seq_len))
        """
        wav = load_audio(
            wav_path,
            sampling_rate=self.config["sample_rate"],
            volume_normalize=self.config["volume_normalize"],
        )

        wav_ref = self.get_ref_clip(wav)

        return wav, wav_ref

    def extract_wav2vec2_features(self, wavs: np.ndarray | List[np.ndarray]) -> torch.Tensor:
        """extract wav2vec2 features
        return: torch.Tensor shape:(batch_size, features_seq_len, feature_dim)
        """
        inputs = self.processor(
            wavs,
            sampling_rate=16000,
            return_tensors="pt",
            padding=True,
            # output_hidden_states=True,
        ).to(self.feature_extractor.device)
        feat = self.feature_extractor(**inputs)
        feats_mix = (feat.hidden_states[11] + feat.hidden_states[14] + feat.hidden_states[16]) / 3

        return feats_mix.detach()

    def tokenize_batch(self, batch: Dict[str, Any]) -> torch.Tensor:
        """tokenize the batch of audio

        Args:
            batch:
                wavs (List[np.ndarray]): batch of audio
                ref_wavs (torch.Tensor): reference audio. shape: (batch_size, seq_len)

        Returns:
            semantic_tokens: semantic tokens. shape: (batch_size, seq_len, latent_dim)
            global_tokens: global tokens. shape: (batch_size, seq_len, global_dim)
        """
        feats = self.extract_wav2vec2_features(batch["wav"])
        batch["feat"] = feats
        semantic_tokens, global_tokens = self.model.tokenize(batch)

        return global_tokens, semantic_tokens

    def batch_tokenize(
        self, audio_paths: Union[str | List[str]]
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        return (global_tokens, semantic_tokens):
        - semantic_tokens: semantic tokens. shape: (batch_size, latent_dim)
        - global_tokens: global tokens. shape: (batch_size, channel, global_dim)
        """
        if isinstance(audio_paths, str):
            audio_paths = [audio_paths]
        wav_list = []
        audio_clip = []
        for audio_path in audio_paths:
            wav, wav_ref = self.process_audio(audio_path)
            wav_list.append(wav)
            audio_clip.append(torch.from_numpy(wav_ref))

        audio_clip = torch.stack(audio_clip).to(self.device)
        audio_features = self.extract_wav2vec2_features(wav_list)

        batch = {
            "ref_wav": audio_clip.float().to(self.device),  # [batch_size,seq_len]
            "feat": audio_features.to(self.device),  # [batch_size,features_seq_len,feature_dim]
        }
        semantic_tokens, global_tokens = self.model.tokenize(batch)  # [batch_size,seq_len]

        if self.device.type == "cuda":
            torch.cuda.empty_cache()

        return global_tokens, semantic_tokens

    def batch_detokenize(
        self, global_tokens: torch.Tensor, semantic_tokens: torch.Tensor
    ) -> np.array:
        wav_rec = self.model.detokenize(semantic_tokens, global_tokens)
        if self.device.type == "cuda":
            torch.cuda.empty_cache()
        return wav_rec.squeeze().cpu().numpy()

    def tokenize(self, audio_path: str) -> Tuple[torch.Tensor, torch.Tensor]:
        """tokenize the audio
        return (global_tokens, semantic_tokens):
        - semantic_tokens: semantic tokens. shape: (batch_size, latent_dim)
        - global_tokens: global tokens. shape: (batch_size, channel, global_dim)
        """
        wav, ref_wav = self.process_audio(audio_path)
        feat = self.extract_wav2vec2_features(wav)
        batch = {
            # "wav": torch.from_numpy(wav).unsqueeze(0).float().to(self.device),
            "ref_wav": torch.from_numpy(ref_wav).unsqueeze(0).float().to(self.device),
            "feat": feat.to(self.device),
        }
        semantic_tokens, global_tokens = self.model.tokenize(batch)

        if self.device.type == "cuda":
            torch.cuda.empty_cache()

        return global_tokens, semantic_tokens

    def detokenize(self, global_tokens: torch.Tensor, semantic_tokens: torch.Tensor) -> np.array:
        """detokenize the tokens to waveform

        Args:
            global_tokens: global tokens. shape: (batch_size, global_dim)
            semantic_tokens: semantic tokens. shape: (batch_size, latent_dim)

        Returns:
            wav_rec: waveform. shape: (batch_size, seq_len) for batch or (seq_len,) for single
        """
        global_tokens = global_tokens.unsqueeze(1)
        wav_rec = self.model.detokenize(semantic_tokens, global_tokens)
        if self.device.type == "cuda":
            torch.cuda.empty_cache()
        return wav_rec.squeeze().cpu().numpy()


# test
if __name__ == "__main__":
    import soundfile as sf
    import os
    from time import perf_counter

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = BiCodecTokenizer(
        model_dir=os.getenv("MODEL_DIR", "pretrained_models/Spark-TTS-0.5B"),
        device=device,
    )

    wav_cases = {
        "single": "example/prompt_audio.wav",
        "multi": ["example/prompt_audio.wav", "example/prompt_audio.wav"],
    }
    for case, wav_path in wav_cases.items():
        start_time = perf_counter()
        if isinstance(wav_path, list):
            global_tokens, semantic_tokens = tokenizer.batch_tokenize(wav_path)
        else:
            global_tokens, semantic_tokens = tokenizer.tokenize(wav_path)
        print(f"""{case} encode elapsed time: {perf_counter()-start_time:.4f} seconds""")
        print(semantic_tokens.shape, global_tokens.shape)

        start_time = perf_counter()
        wav_rec = tokenizer.detokenize(global_tokens.squeeze(1), semantic_tokens)
        print(f"""{case} decode elapsed time: {perf_counter()-start_time:.4f} seconds""")
        print(wav_rec.shape)
        if len(wav_rec.shape) > 1:
            for i, wav in enumerate(wav_rec):
                sf.write(f"example/prompt_recon_{i}.wav", wav, 16000)
        if len(wav_rec.shape) == 1:
            sf.write("example/prompt_recon.wav", wav_rec, 16000)
