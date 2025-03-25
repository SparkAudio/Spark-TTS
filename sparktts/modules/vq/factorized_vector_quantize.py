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

# Heavily based on https://github.com/lucidrains/vector-quantize-pytorch


from typing import Any, Dict

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from torch.nn.utils import parametrizations


def WNConv1d(*args, **kwargs):
    return parametrizations.weight_norm(nn.Conv1d(*args, **kwargs))


def ema_inplace(moving_avg, new, decay):
    moving_avg.data.mul_(decay).add_(new, alpha=(1 - decay))


class FactorizedVectorQuantize(nn.Module):
    def __init__(
        self,
        input_dim: int,
        codebook_size: int,
        codebook_dim: int,
        commitment: float,
        codebook_loss_weight: float = 1.0,
        decay: float = 0.99,
        threshold_ema_dead_code: float = 2,
        momentum: float = 0.99,
        **kwargs,
    ):
        super().__init__()
        self.input_dim = input_dim
        self.codebook_size = codebook_size
        self.codebook_dim = codebook_dim
        self.commitment = commitment
        self.codebook_loss_weight = codebook_loss_weight
        self.decay = decay
        self.threshold_ema_dead_code = threshold_ema_dead_code
        self.momentum = momentum

        if input_dim != self.codebook_dim:
            self.in_project = WNConv1d(input_dim, self.codebook_dim, kernel_size=1)
            self.out_project = WNConv1d(self.codebook_dim, input_dim, kernel_size=1)

        else:
            self.in_project = nn.Identity()
            self.out_project = nn.Identity()

        self.codebook = nn.Embedding(self.codebook_size, self.codebook_dim)
        self.register_buffer("cluster_size", torch.zeros(self.codebook_size))

    def forward(self, z: torch.Tensor) -> Dict[str, Any]:
        """Quantized the input tensor using a fixed codebook and returns
        the corresponding codebook vectors

        Parameters
        ----------
        z : Tensor[B x D x T]

        Returns
        -------
        Tensor[B x D x T]
            Quantized continuous representation of input
        Tensor[1]
            Commitment loss to train encoder to predict vectors closer to codebook
            entries
        Tensor[1]
            Codebook loss to update the codebook
        Tensor[B x T]
            Codebook indices (quantized discrete representation of input)
        Tensor[B x D x T]
            Projected latents (continuous representation of input before quantization)
        """
        # transpose since we use linear

        # Factorized codes project input into low-dimensional space if self.input_dim != self.codebook_dim
        z_e = self.in_project(z)
        z_q, indices, dists = self.decode_latents(z_e)

        # statistic the usage of codes
        embed_onehot = F.one_hot(indices, self.codebook_size).type(z_e.dtype)
        avg_probs = torch.mean(embed_onehot.reshape(-1, self.codebook_size), dim=0)
        perplexity = torch.exp(-torch.sum(avg_probs * torch.log(avg_probs + 1e-10)))

        active_num = (embed_onehot.sum(0).sum(0) > 0).sum()
        if self.training:
            # We do the expiry of code at that point as buffers are in sync
            # and all the workers will take the same decision.
            ema_inplace(self.cluster_size, embed_onehot.sum(0).sum(0), self.decay)
            active_num = sum(self.cluster_size > self.threshold_ema_dead_code)

        if self.training:
            commit_loss = (
                F.mse_loss(z_e, z_q.detach(), reduction="none").mean([1, 2])
                * self.commitment
            )

            codebook_loss = (
                F.mse_loss(z_q, z_e.detach(), reduction="none").mean([1, 2])
                * self.codebook_loss_weight
            )

        else:
            commit_loss = torch.zeros(0, device=z.device)
            codebook_loss = torch.zeros(0, device=z.device)

        z_q = (
            z_e + (z_q - z_e).detach()
        )  # noop in forward pass, straight-through gradient estimator in backward pass

        z_q = self.out_project(z_q)

        vq_loss = (commit_loss + codebook_loss).mean()

        return {
            "z_q": z_q,
            "indices": indices,
            "dists": dists,
            "vq_loss": vq_loss,
            "perplexity": perplexity,
            "active_num": active_num.float(),
        }

    def vq2emb(self, vq, out_proj=True):
        emb = self.embed_code(vq)
        if out_proj:
            emb = self.out_project(emb)
        return emb

    def tokenize(self, z: torch.Tensor) -> torch.Tensor:
        """tokenize the input tensor"""
        z_e = self.in_project(z)
        _, indices, _ = self.decode_latents(z_e)
        return indices

    def detokenize(self, indices):
        """detokenize the input indices"""
        z_q = self.decode_code(indices)
        z_q = self.out_project(z_q)
        return z_q

    def get_emb(self):
        return self.codebook.weight

    def embed_code(self, embed_id):
        return F.embedding(embed_id, self.codebook.weight)

    def decode_code(self, embed_id):
        return self.embed_code(embed_id).transpose(1, 2)

    def decode_latents(self, latents):
        encodings = rearrange(latents, "b d t -> (b t) d")
        codebook = self.codebook.weight

        # L2 normalize encodings and codebook
        encodings = F.normalize(encodings)
        codebook = F.normalize(codebook)

        # Compute euclidean distance between encodings and codebook,
        # with L2 normalization, the distance is equal to cosine distance
        dist = (
            encodings.pow(2).sum(1, keepdim=True)
            - 2 * encodings @ codebook.t()
            + codebook.pow(2).sum(1, keepdim=True).t()
        )
        indices = rearrange((-dist).max(1)[1], "(b t) -> b t", b=latents.size(0))
        z_q = self.decode_code(indices)

        return z_q, indices, dist
