# Copyright 2025, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions
# are met:
#  * Redistributions of source code must retain the above copyright
#    notice, this list of conditions and the following disclaimer.
#  * Redistributions in binary form must reproduce the above copyright
#    notice, this list of conditions and the following disclaimer in the
#    documentation and/or other materials provided with the distribution.
#  * Neither the name of NVIDIA CORPORATION nor the names of its
#    contributors may be used to endorse or promote products derived
#    from this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS ``AS IS'' AND ANY
# EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
# PURPOSE ARE DISCLAIMED.  IN NO EVENT SHALL THE COPYRIGHT OWNER OR
# CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
# EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
# PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
# PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY
# OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
# (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
import json
import torch
from torch.utils.dlpack import to_dlpack

import triton_python_backend_utils as pb_utils

import os
import numpy as np

from sparktts.models.audio_tokenizer import BiCodecTokenizer

class TritonPythonModel:
    """Triton Python model for audio tokenization.
    
    This model takes reference audio input and extracts semantic and global tokens
    using BiCodec tokenizer.
    """

    def initialize(self, args):
        """Initialize the model.
        
        Args:
            args: Dictionary containing model configuration
        """
        # Parse model parameters
        parameters = json.loads(args['model_config'])['parameters']
        model_params = {k: v["string_value"] for k, v in parameters.items()}
        
        # Initialize tokenizer
        self.device = torch.device("cuda")
        self.audio_tokenizer = BiCodecTokenizer(model_params["model_dir"], 
                                              device=self.device)

    def get_ref_clip(self, wav: np.ndarray) -> np.ndarray:
        """Extract reference audio clip for speaker embedding.
        
        Args:
            wav: Input waveform array
            
        Returns:
            Reference clip of fixed duration
        """
        SAMPLE_RATE = 16000
        REF_SEGMENT_DURATION = 6  # seconds
        LATENT_HOP_LENGTH = 320

        ref_segment_length = (
            int(SAMPLE_RATE * REF_SEGMENT_DURATION)
            // LATENT_HOP_LENGTH
            * LATENT_HOP_LENGTH
        )
        wav_length = len(wav)

        if ref_segment_length > wav_length:
            # Repeat and truncate if input is too short
            repeat_times = ref_segment_length // wav_length + 1
            wav = np.tile(wav, repeat_times)

        return wav[:ref_segment_length]

    def execute(self, requests):
        """Execute inference on the batched requests.
        
        Args:
            requests: List of inference requests
            
        Returns:
            List of inference responses containing tokenized outputs
        """
        reference_wav_list = []
        reference_wav_ref_clip_list = []

        # Process each request in batch
        for request in requests:
            # Extract input tensors
            wav_array = pb_utils.get_input_tensor_by_name(
                request, "reference_wav").as_numpy()
            wav_len = pb_utils.get_input_tensor_by_name(
                request, "reference_wav_len").as_numpy().item()

            # Prepare inputs
            wav = wav_array[:, :wav_len].squeeze(0)
            reference_wav_list.append(wav)
            
            wav_ref_clip = self.get_ref_clip(wav)
            reference_wav_ref_clip_list.append(torch.from_numpy(wav_ref_clip))

        # Batch process through tokenizer
        ref_wav_clip_tensor = torch.stack(reference_wav_ref_clip_list, dim=0)
        wav2vec2_features = self.audio_tokenizer.extract_wav2vec2_features(
            reference_wav_list)
        
        audio_tokenizer_input = {
            "ref_wav": ref_wav_clip_tensor.to(self.device),
            "feat": wav2vec2_features.to(self.device),
        }
        semantic_tokens, global_tokens = self.audio_tokenizer.model.tokenize(
            audio_tokenizer_input)

        # Prepare responses
        responses = []
        for i in range(len(requests)):
            global_tokens_tensor = pb_utils.Tensor.from_dlpack(
                "global_tokens", to_dlpack(global_tokens[i]))
            semantic_tokens_tensor = pb_utils.Tensor.from_dlpack(
                "semantic_tokens", to_dlpack(semantic_tokens[i]))
            
            inference_response = pb_utils.InferenceResponse(
                output_tensors=[global_tokens_tensor, semantic_tokens_tensor])
            responses.append(inference_response)
                             
        return responses
