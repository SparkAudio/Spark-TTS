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
import torch.nn as nn

from typing import Tuple
from torch.nn.utils import parametrizations

from typing import Optional


class ConvNeXtBlock(nn.Module):
    """ConvNeXt Block adapted from https://github.com/facebookresearch/ConvNeXt to 1D audio signal.

    Args:
        dim (int): Number of input channels.
        intermediate_dim (int): Dimensionality of the intermediate layer.
        layer_scale_init_value (float, optional): Initial value for the layer scale. None means no scaling.
            Defaults to None.
        adanorm_num_embeddings (int, optional): Number of embeddings for AdaLayerNorm.
            None means non-conditional LayerNorm. Defaults to None.
    """

    def __init__(
        self,
        dim: int,
        intermediate_dim: int,
        layer_scale_init_value: float,
        condition_dim: Optional[int] = None,
    ):
        super().__init__()
        self.dwconv = nn.Conv1d(
            dim, dim, kernel_size=7, padding=3, groups=dim
        )  # depthwise conv
        self.adanorm = condition_dim is not None
        if condition_dim:
            self.norm = AdaLayerNorm(condition_dim, dim, eps=1e-6)
        else:
            self.norm = nn.LayerNorm(dim, eps=1e-6)
        self.pwconv1 = nn.Linear(
            dim, intermediate_dim
        )  # pointwise/1x1 convs, implemented with linear layers
        self.act = nn.GELU()
        self.pwconv2 = nn.Linear(intermediate_dim, dim)
        self.gamma = (
            nn.Parameter(layer_scale_init_value * torch.ones(dim), requires_grad=True)
            if layer_scale_init_value > 0
            else None
        )

    def forward(
        self, x: torch.Tensor, cond_embedding_id: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        residual = x
        x = self.dwconv(x)
        x = x.transpose(1, 2)  # (B, C, T) -> (B, T, C)
        if self.adanorm:
            assert cond_embedding_id is not None
            x = self.norm(x, cond_embedding_id)
        else:
            x = self.norm(x)
        x = self.pwconv1(x)
        x = self.act(x)
        x = self.pwconv2(x)
        if self.gamma is not None:
            x = self.gamma * x
        x = x.transpose(1, 2)  # (B, T, C) -> (B, C, T)

        x = residual + x
        return x


class AdaLayerNorm(nn.Module):
    """
    Adaptive Layer Normalization module with learnable embeddings per `num_embeddings` classes

    Args:
        condition_dim (int): Dimension of the condition.
        embedding_dim (int): Dimension of the embeddings.
    """

    def __init__(self, condition_dim: int, embedding_dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.dim = embedding_dim
        self.scale = nn.Linear(condition_dim, embedding_dim)
        self.shift = nn.Linear(condition_dim, embedding_dim)
        torch.nn.init.ones_(self.scale.weight)
        torch.nn.init.zeros_(self.shift.weight)

    def forward(self, x: torch.Tensor, cond_embedding: torch.Tensor) -> torch.Tensor:
        scale = self.scale(cond_embedding)
        shift = self.shift(cond_embedding)
        x = nn.functional.layer_norm(x, (self.dim,), eps=self.eps)
        x = x * scale.unsqueeze(1) + shift.unsqueeze(1)
        return x


class ResBlock1(nn.Module):
    """
    ResBlock adapted from HiFi-GAN V1 (https://github.com/jik876/hifi-gan) with dilated 1D convolutions,
    but without upsampling layers.

    Args:
        dim (int): Number of input channels.
        kernel_size (int, optional): Size of the convolutional kernel. Defaults to 3.
        dilation (tuple[int], optional): Dilation factors for the dilated convolutions.
            Defaults to (1, 3, 5).
        lrelu_slope (float, optional): Negative slope of the LeakyReLU activation function.
            Defaults to 0.1.
        layer_scale_init_value (float, optional): Initial value for the layer scale. None means no scaling.
            Defaults to None.
    """

    def __init__(
        self,
        dim: int,
        kernel_size: int = 3,
        dilation: Tuple[int, int, int] = (1, 3, 5),
        lrelu_slope: float = 0.1,
        layer_scale_init_value: Optional[float] = None,
    ):
        super().__init__()
        self.lrelu_slope = lrelu_slope
        self.convs1 = nn.ModuleList(
            [
                parametrizations.weight_norm(
                    nn.Conv1d(
                        dim,
                        dim,
                        kernel_size,
                        1,
                        dilation=dilation[0],
                        padding=self.get_padding(kernel_size, dilation[0]),
                    )
                ),
                parametrizations.weight_norm(
                    nn.Conv1d(
                        dim,
                        dim,
                        kernel_size,
                        1,
                        dilation=dilation[1],
                        padding=self.get_padding(kernel_size, dilation[1]),
                    )
                ),
                parametrizations.weight_norm(
                    nn.Conv1d(
                        dim,
                        dim,
                        kernel_size,
                        1,
                        dilation=dilation[2],
                        padding=self.get_padding(kernel_size, dilation[2]),
                    )
                ),
            ]
        )

        self.convs2 = nn.ModuleList(
            [
                parametrizations.weight_norm(
                    nn.Conv1d(
                        dim,
                        dim,
                        kernel_size,
                        1,
                        dilation=1,
                        padding=self.get_padding(kernel_size, 1),
                    )
                ),
                parametrizations.weight_norm(
                    nn.Conv1d(
                        dim,
                        dim,
                        kernel_size,
                        1,
                        dilation=1,
                        padding=self.get_padding(kernel_size, 1),
                    )
                ),
                parametrizations.weight_norm(
                    nn.Conv1d(
                        dim,
                        dim,
                        kernel_size,
                        1,
                        dilation=1,
                        padding=self.get_padding(kernel_size, 1),
                    )
                ),
            ]
        )

        self.gamma = nn.ParameterList(
            [
                (
                    nn.Parameter(
                        layer_scale_init_value * torch.ones(dim, 1), requires_grad=True
                    )
                    if layer_scale_init_value is not None
                    else None
                ),
                (
                    nn.Parameter(
                        layer_scale_init_value * torch.ones(dim, 1), requires_grad=True
                    )
                    if layer_scale_init_value is not None
                    else None
                ),
                (
                    nn.Parameter(
                        layer_scale_init_value * torch.ones(dim, 1), requires_grad=True
                    )
                    if layer_scale_init_value is not None
                    else None
                ),
            ]
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for c1, c2, gamma in zip(self.convs1, self.convs2, self.gamma):
            xt = torch.nn.functional.leaky_relu(x, negative_slope=self.lrelu_slope)
            xt = c1(xt)
            xt = torch.nn.functional.leaky_relu(xt, negative_slope=self.lrelu_slope)
            xt = c2(xt)
            if gamma is not None:
                xt = gamma * xt
            x = xt + x
        return x

    def remove_weight_norm(self):
        for l in self.convs1:
            parametrizations.remove_weight_norm(l)
        for l in self.convs2:
            parametrizations.remove_weight_norm(l)

    @staticmethod
    def get_padding(kernel_size: int, dilation: int = 1) -> int:
        return int((kernel_size * dilation - dilation) / 2)


class Backbone(nn.Module):
    """Base class for the generator's backbone. It preserves the same temporal resolution across all layers."""

    def forward(self, x: torch.Tensor, **kwargs) -> torch.Tensor:
        """
        Args:
            x (Tensor): Input tensor of shape (B, C, L), where B is the batch size,
                        C denotes output features, and L is the sequence length.

        Returns:
            Tensor: Output of shape (B, L, H), where B is the batch size, L is the sequence length,
                    and H denotes the model dimension.
        """
        raise NotImplementedError("Subclasses must implement the forward method.")


class VocosBackbone(Backbone):
    """
    Vocos backbone module built with ConvNeXt blocks. Supports additional conditioning with Adaptive Layer Normalization

    Args:
        input_channels (int): Number of input features channels.
        dim (int): Hidden dimension of the model.
        intermediate_dim (int): Intermediate dimension used in ConvNeXtBlock.
        num_layers (int): Number of ConvNeXtBlock layers.
        layer_scale_init_value (float, optional): Initial value for layer scaling. Defaults to `1 / num_layers`.
        adanorm_num_embeddings (int, optional): Number of embeddings for AdaLayerNorm.
                                                None means non-conditional model. Defaults to None.
    """

    def __init__(
        self,
        input_channels: int,
        dim: int,
        intermediate_dim: int,
        num_layers: int,
        layer_scale_init_value: Optional[float] = None,
        condition_dim: Optional[int] = None,
    ):
        super().__init__()
        self.input_channels = input_channels
        self.embed = nn.Conv1d(input_channels, dim, kernel_size=7, padding=3)
        self.adanorm = condition_dim is not None
        if condition_dim:
            self.norm = AdaLayerNorm(condition_dim, dim, eps=1e-6)
        else:
            self.norm = nn.LayerNorm(dim, eps=1e-6)
        layer_scale_init_value = layer_scale_init_value or 1 / num_layers
        self.convnext = nn.ModuleList(
            [
                ConvNeXtBlock(
                    dim=dim,
                    intermediate_dim=intermediate_dim,
                    layer_scale_init_value=layer_scale_init_value,
                    condition_dim=condition_dim,
                )
                for _ in range(num_layers)
            ]
        )
        self.final_layer_norm = nn.LayerNorm(dim, eps=1e-6)
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, (nn.Conv1d, nn.Linear)):
            nn.init.trunc_normal_(m.weight, std=0.02)
            nn.init.constant_(m.bias, 0)

    def forward(self, x: torch.Tensor, condition: torch.Tensor = None) -> torch.Tensor:
        x = self.embed(x)
        if self.adanorm:
            assert condition is not None
            x = self.norm(x.transpose(1, 2), condition)
        else:
            x = self.norm(x.transpose(1, 2))
        x = x.transpose(1, 2)
        for conv_block in self.convnext:
            x = conv_block(x, condition)
        x = self.final_layer_norm(x.transpose(1, 2))
        return x


class VocosResNetBackbone(Backbone):
    """
    Vocos backbone module built with ResBlocks.

    Args:
        input_channels (int): Number of input features channels.
        dim (int): Hidden dimension of the model.
        num_blocks (int): Number of ResBlock1 blocks.
        layer_scale_init_value (float, optional): Initial value for layer scaling. Defaults to None.
    """

    def __init__(
        self,
        input_channels,
        dim,
        num_blocks,
        layer_scale_init_value=None,
    ):
        super().__init__()
        self.input_channels = input_channels
        self.embed = parametrizations.weight_norm(
            nn.Conv1d(input_channels, dim, kernel_size=3, padding=1)
        )
        layer_scale_init_value = layer_scale_init_value or 1 / num_blocks / 3
        self.resnet = nn.Sequential(
            *[
                ResBlock1(dim=dim, layer_scale_init_value=layer_scale_init_value)
                for _ in range(num_blocks)
            ]
        )

    def forward(self, x: torch.Tensor, **kwargs) -> torch.Tensor:
        x = self.embed(x)
        x = self.resnet(x)
        x = x.transpose(1, 2)
        return x
