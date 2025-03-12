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
import os
import logging
from typing import List, Dict

import torch
from torch.utils.dlpack import to_dlpack

import triton_python_backend_utils as pb_utils

from sparktts.models.bicodec import BiCodec

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class TritonPythonModel:
    """Triton Python model for vocoder.
    
    This model takes global and semantic tokens as input and generates audio waveforms
    using the BiCodec vocoder.
    """

    def initialize(self, args):
        """Initialize the model.
        
        Args:
            args: Dictionary containing model configuration
        """
        # Parse model parameters
        parameters = json.loads(args['model_config'])['parameters']
        model_params = {key: value["string_value"] for key, value in parameters.items()}
        model_dir = model_params["model_dir"]
        
        # Initialize device and vocoder
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logger.info(f"Initializing vocoder from {model_dir} on {self.device}")
        
        self.vocoder = BiCodec.load_from_checkpoint(f"{model_dir}/BiCodec")
        del self.vocoder.encoder, self.vocoder.postnet
        self.vocoder.eval().to(self.device)  # Set model to evaluation mode

        logger.info("Vocoder initialized successfully")


    def execute(self, requests):
        """Execute inference on the batched requests.
        
        Args:
            requests: List of inference requests
            
        Returns:
            List of inference responses containing generated waveforms
        """
        global_tokens_list, semantic_tokens_list = [], []

        # Process each request in batch
        for request in requests:
            global_tokens_tensor = pb_utils.get_input_tensor_by_name(request, "global_tokens").as_numpy()
            semantic_tokens_tensor = pb_utils.get_input_tensor_by_name(request, "semantic_tokens").as_numpy()
            global_tokens_list.append(torch.from_numpy(global_tokens_tensor).to(self.device))
            semantic_tokens_list.append(torch.from_numpy(semantic_tokens_tensor).to(self.device))

        # Concatenate tokens for batch processing
        global_tokens = torch.cat(global_tokens_list, dim=0)
        semantic_tokens = torch.cat(semantic_tokens_list, dim=0)
        

        # Generate waveforms
        with torch.no_grad():
            wavs = self.vocoder.detokenize(semantic_tokens, global_tokens.unsqueeze(1))

        # Prepare responses
        responses = []
        for i in range(len(requests)):
            wav_tensor = pb_utils.Tensor.from_dlpack("waveform", to_dlpack(wavs[i]))
            inference_response = pb_utils.InferenceResponse(output_tensors=[wav_tensor])
            responses.append(inference_response)
                             
        return responses
