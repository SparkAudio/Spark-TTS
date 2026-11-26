import random
from math import ceil
import torch
import torch.nn.functional as F
import torch.distributed as dist

from typing import List
from torch import nn
from torch.nn import Module
from torch.amp import autocast
from einx import get_at
from einops import rearrange, reduce, pack, unpack

from sparktts.modules.fsq.finite_scalar_quantization import FSQ


def exists(val):
    return val is not None


def first(l):
    return l[0]


def default(val, d):
    return val if exists(val) else d


def round_up_multiple(num, mult):
    return ceil(num / mult) * mult


# distributed helpers


def is_distributed():
    return dist.is_initialized() and dist.get_world_size() > 1


def get_maybe_sync_seed(device, max_size=10_000):
    rand_int = torch.randint(0, max_size, (), device=device)

    if is_distributed():
        dist.all_reduce(rand_int)

    return rand_int.item()


class ResidualFSQ(Module):
    """Follows Algorithm 1. in https://arxiv.org/pdf/2107.03312.pdf"""

    def __init__(
        self,
        *,
        levels: List[int],
        num_quantizers,
        dim=None,
        is_channel_first=False,
        quantize_dropout=False,
        quantize_dropout_cutoff_index=0,
        quantize_dropout_multiple_of=1,
        **kwargs,
    ):
        super().__init__()
        codebook_dim = len(levels)
        dim = default(dim, codebook_dim)

        requires_projection = codebook_dim != dim
        self.project_in = (
            nn.Linear(dim, codebook_dim) if requires_projection else nn.Identity()
        )
        self.project_out = (
            nn.Linear(codebook_dim, dim) if requires_projection else nn.Identity()
        )
        self.has_projections = requires_projection

        self.is_channel_first = is_channel_first
        self.num_quantizers = num_quantizers

        self.levels = levels
        self.layers = nn.ModuleList([])

        levels_tensor = torch.Tensor(levels)

        scales = []

        for ind in range(num_quantizers):
            scales.append((levels_tensor - 1) ** -ind)

            fsq = FSQ(levels=levels, dim=codebook_dim, **kwargs)

            self.layers.append(fsq)

        assert all([not fsq.has_projections for fsq in self.layers])

        self.codebook_size = self.layers[0].codebook_size

        self.register_buffer("scales", torch.stack(scales), persistent=False)

        self.quantize_dropout = quantize_dropout and num_quantizers > 1

        assert quantize_dropout_cutoff_index >= 0

        self.quantize_dropout_cutoff_index = quantize_dropout_cutoff_index
        self.quantize_dropout_multiple_of = quantize_dropout_multiple_of  # encodec paper proposes structured dropout, believe this was set to 4

    @property
    def codebooks(self):
        codebooks = [layer.implicit_codebook for layer in self.layers]
        codebooks = torch.stack(codebooks, dim=0)
        return codebooks

    def get_codes_from_indices(self, indices):

        batch, quantize_dim = indices.shape[0], indices.shape[-1]

        # may also receive indices in the shape of 'b h w q' (accept_image_fmap)

        indices, ps = pack([indices], "b * q")

        # because of quantize dropout, one can pass in indices that are coarse
        # and the network should be able to reconstruct

        if quantize_dim < self.num_quantizers:
            assert (
                self.quantize_dropout > 0.0
            ), "quantize dropout must be greater than 0 if you wish to reconstruct from a signal with less fine quantizations"
            indices = F.pad(indices, (0, self.num_quantizers - quantize_dim), value=-1)

        # take care of quantizer dropout

        mask = indices == -1
        indices = indices.masked_fill(
            mask, 0
        )  # have it fetch a dummy code to be masked out later

        all_codes = get_at("q [c] d, b n q -> q b n d", self.codebooks, indices)

        # mask out any codes that were dropout-ed

        all_codes = all_codes.masked_fill(rearrange(mask, "b n q -> q b n 1"), 0.0)

        # scale the codes

        scales = rearrange(self.scales, "q d -> q 1 1 d")
        all_codes = all_codes * scales

        # if (accept_image_fmap = True) then return shape (quantize, batch, height, width, dimension)

        (all_codes,) = unpack(all_codes, ps, "q b * d")

        return all_codes

    def get_output_from_indices(self, indices):
        codes = self.get_codes_from_indices(indices)
        codes_summed = reduce(codes, "q ... -> ...", "sum")
        return self.project_out(codes_summed)

    def forward(self, x, return_all_codes=False, rand_quantize_dropout_fixed_seed=None):
        num_quant, quant_dropout_multiple_of, device = (
            self.num_quantizers,
            self.quantize_dropout_multiple_of,
            x.device,
        )

        # handle channel first

        if self.is_channel_first:
            x = rearrange(x, "b d ... -> b ... d")
            x, ps = pack([x], "b * d")

        # maybe project in

        x = self.project_in(x)

        quantized_out = 0.0
        residual = x

        all_indices = []

        should_quantize_dropout = self.training and self.quantize_dropout

        # sample a layer index at which to dropout further residual quantization
        # also prepare null indices

        if should_quantize_dropout:

            # check if seed is manually passed in

            if not exists(rand_quantize_dropout_fixed_seed):
                rand_quantize_dropout_fixed_seed = get_maybe_sync_seed(device)

            rand = random.Random(rand_quantize_dropout_fixed_seed)

            rand_quantize_dropout_index = rand.randrange(
                self.quantize_dropout_cutoff_index, num_quant
            )

            if quant_dropout_multiple_of != 1:
                rand_quantize_dropout_index = (
                    round_up_multiple(
                        rand_quantize_dropout_index + 1, quant_dropout_multiple_of
                    )
                    - 1
                )

            null_indices = torch.full(
                x.shape[:2], -1.0, device=device, dtype=torch.long
            )

        # go through the layers

        with autocast("cuda", enabled=False):
            for quantizer_index, (layer, scale) in enumerate(
                zip(self.layers, self.scales)
            ):

                if (
                    should_quantize_dropout
                    and quantizer_index > rand_quantize_dropout_index
                ):
                    all_indices.append(null_indices)
                    continue

                quantized, indices = layer(residual / scale)

                quantized = quantized * scale

                residual = residual - quantized.detach()
                quantized_out = quantized_out + quantized

                all_indices.append(indices)

        # project out, if needed

        quantized_out = self.project_out(quantized_out)

        # stack all indices

        all_indices = torch.stack(all_indices, dim=-1)

        # channel first out

        if self.is_channel_first:
            (quantized_out,) = unpack(quantized_out, ps, "b * d")
            (all_indices,) = unpack(all_indices, ps, "b * d")

            quantized_out = rearrange(quantized_out, "b ... d -> b d ...")
            all_indices = rearrange(all_indices, "b ... d -> b d ...")

        # return

        ret = (quantized_out, all_indices)

        if not return_all_codes:
            return ret

        # whether to return all codes from all codebooks across layers

        all_codes = self.get_codes_from_indices(all_indices)

        # will return all codes in shape (quantizer, batch, sequence length, codebook dimension)

        return (*ret, all_codes)


# grouped residual fsq


class GroupedResidualFSQ(Module):
    def __init__(self, *, dim, groups=1, accept_image_fmap=False, **kwargs):
        super().__init__()
        self.dim = dim
        self.groups = groups
        assert (dim % groups) == 0
        dim_per_group = dim // groups

        self.accept_image_fmap = accept_image_fmap

        self.rvqs = nn.ModuleList([])

        for _ in range(groups):
            self.rvqs.append(ResidualFSQ(dim=dim_per_group, **kwargs))

        self.codebook_size = self.rvqs[0].codebook_size

    @property
    def codebooks(self):
        return torch.stack(tuple(rvq.codebooks for rvq in self.rvqs))

    @property
    def split_dim(self):
        return 1 if self.accept_image_fmap else -1

    def get_codes_from_indices(self, indices):
        codes = tuple(
            rvq.get_codes_from_indices(chunk_indices)
            for rvq, chunk_indices in zip(self.rvqs, indices)
        )
        return torch.stack(codes)

    def get_output_from_indices(self, indices):
        outputs = tuple(
            rvq.get_output_from_indices(chunk_indices)
            for rvq, chunk_indices in zip(self.rvqs, indices)
        )
        return torch.cat(outputs, dim=self.split_dim)

    def forward(self, x, return_all_codes=False):
        shape, split_dim, device = x.shape, self.split_dim, x.device
        assert shape[split_dim] == self.dim

        # split the feature dimension into groups

        x = x.chunk(self.groups, dim=split_dim)

        forward_kwargs = dict(
            return_all_codes=return_all_codes,
            rand_quantize_dropout_fixed_seed=(
                get_maybe_sync_seed(device) if self.training else None
            ),
        )

        # invoke residual vq on each group

        out = tuple(rvq(chunk, **forward_kwargs) for rvq, chunk in zip(self.rvqs, x))
        out = tuple(zip(*out))

        # otherwise, get all the zipped outputs and combine them

        quantized, all_indices, *maybe_all_codes = out

        quantized = torch.cat(quantized, dim=split_dim)
        all_indices = torch.stack(all_indices)

        ret = (quantized, all_indices, *maybe_all_codes)
        return ret


if __name__ == "__main__":
    model = ResidualFSQ(
        levels=[4, 4, 4, 4, 4, 4],
        num_quantizers=1,
        dim=30,
        is_channel_first=True,
        quantize_dropout=False,
    )
    x = torch.randn(2, 30, 10)
    quantize, embed_ind = model(x)

    emb_from_ind = model.get_output_from_indices(embed_ind.transpose(1, 2))

    print(quantize == emb_from_ind.transpose(1, 2))

    print("quantize shape", quantize.shape)
    print("embed_ind", embed_ind)
