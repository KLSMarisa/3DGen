from abc import abstractmethod
import math
from typing import Any
import numpy as np
import torch as th
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange, repeat

from .util import (
    checkpoint,
    conv_nd,
    linear,
    avg_pool_nd,
    zero_module,
    normalization,
    timestep_embedding,
    exists
)
from .attention import SpatialTransformer, TemporalTransformer, TriplaneTransformer


# dummy replace
def convert_module_to_f16(x):
    pass

def convert_module_to_f32(x):
    pass


class TimestepBlock(nn.Module):
    """
    Any module where forward() takes timestep embeddings as a second argument.
    """

    @abstractmethod
    def forward(self, x, emb):
        """
        Apply the module to `x` given `emb` timestep embeddings.
        """


class TimestepEmbedSequential(nn.Sequential, TimestepBlock):
    """
    A sequential module that passes timestep embeddings to the children that
    support it as an extra input.
    """

    def forward(self, x, emb, context=None, dataset=None):
        for layer in self:
            if isinstance(layer, TimestepBlock):
                x = layer(x, emb)
            elif isinstance(layer, SpatialTransformer) or isinstance(layer, TemporalTransformer) or isinstance(layer, TriplaneTransformer) or isinstance(layer, SpatialTemporalTransformer) or isinstance(layer, SpatialTriplaneTransformer):
                x = layer(x, context)
            elif isinstance(layer, NormResBlock_SpatialTransformer_and_UpDownsample) or isinstance(layer, SpatialTemporalNormResBlockTransformer_and_UpDownsample) or isinstance(layer, SpatialTriplaneNormResBlockTransformer_and_UpDownsample):
                x = layer(x, emb, context, dataset)
            else:
                x = layer(x)
        return x


class Upsample(nn.Module):
    """
    An upsampling layer with an optional convolution.
    :param channels: channels in the inputs and outputs.
    :param use_conv: a bool determining if a convolution is applied.
    :param dims: determines if the signal is 1D, 2D, or 3D. If 3D, then
                 upsampling occurs in the inner-two dimensions.
    """

    def __init__(self, channels, use_conv, dims=2, out_channels=None, padding=1):
        super().__init__()
        self.channels = channels
        self.out_channels = out_channels or channels
        self.use_conv = use_conv
        self.dims = dims
        if use_conv:
            self.conv = conv_nd(dims, self.channels, self.out_channels, 3, padding=padding)

    def forward(self, x):
        assert x.shape[1] == self.channels
        B, C, F, H, W = x.shape
        if self.dims == 2:
            x = rearrange(x, 'b c f h w -> (b f) c h w')
        elif self.dims == 1:
            x = rearrange(x, 'b c f h w -> (b h w) c f')
        else:
            raise NotImplementedError
        if self.dims == 3:
            x = th.nn.functional.interpolate(
                x, (x.shape[2], x.shape[3] * 2, x.shape[4] * 2), mode="nearest"
            )
        else:
            x = th.nn.functional.interpolate(x, scale_factor=2, mode="nearest")
        if self.use_conv:
            x = self.conv(x)
        if self.dims == 2:
            x = rearrange(x, '(b f) c h w -> b c f h w', b=B, f=F)
        elif self.dims == 1:
            x = rearrange(x, '(b h w) c f -> b c f h w', b=B, h=H, w=W)
        else:
            raise NotImplementedError
        return x


class Downsample(nn.Module):
    """
    A downsampling layer with an optional convolution.
    :param channels: channels in the inputs and outputs.
    :param use_conv: a bool determining if a convolution is applied.
    :param dims: determines if the signal is 1D, 2D, or 3D. If 3D, then
                 downsampling occurs in the inner-two dimensions.
    """

    def __init__(self, channels, use_conv, dims=2, out_channels=None,padding=1):
        super().__init__()
        self.channels = channels
        self.out_channels = out_channels or channels
        self.use_conv = use_conv
        self.dims = dims
        stride = 2 if dims != 3 else (1, 2, 2)
        if use_conv:
            self.op = conv_nd(
                dims, self.channels, self.out_channels, 3, stride=stride, padding=padding
            )
        else:
            assert self.channels == self.out_channels
            self.op = avg_pool_nd(dims, kernel_size=stride, stride=stride)

    def forward(self, x):
        assert x.shape[1] == self.channels
        B, C, F, H, W = x.shape
        if self.dims == 2:
            x = rearrange(x, 'b c f h w -> (b f) c h w')
        elif self.dims == 1:
            x = rearrange(x, 'b c f h w -> (b h w) c f')
        else:
            raise NotImplementedError
        out = self.op(x)
        if self.dims == 2:
            out = rearrange(out, '(b f) c h w -> b c f h w', b=B, f=F)
        elif self.dims == 1:
            out = rearrange(out, '(b h w) c f -> b c f h w', b=B, h=H, w=W)
        else:
            raise NotImplementedError
        return out


class ResBlock(TimestepBlock):
    """
    A residual block that can optionally change the number of channels.
    :param channels: the number of input channels.
    :param emb_channels: the number of timestep embedding channels.
    :param dropout: the rate of dropout.
    :param out_channels: if specified, the number of out channels.
    :param use_conv: if True and out_channels is specified, use a spatial
        convolution instead of a smaller 1x1 convolution to change the
        channels in the skip connection.
    :param dims: determines if the signal is 1D, 2D, or 3D.
    :param use_checkpoint: if True, use gradient checkpointing on this module.
    :param up: if True, use this block for upsampling.
    :param down: if True, use this block for downsampling.
    """

    def __init__(
        self,
        channels,
        emb_channels,
        dropout,
        out_channels=None,
        use_conv=False,
        use_scale_shift_norm=False,
        dims=2,
        use_checkpoint=False,
        up=False,
        down=False,
    ):
        super().__init__()
        self.channels = channels
        self.emb_channels = emb_channels
        self.dropout = dropout
        self.out_channels = out_channels or channels
        self.use_conv = use_conv
        self.use_checkpoint = use_checkpoint
        self.use_scale_shift_norm = use_scale_shift_norm
        self.dims = dims

        self.in_layers = nn.Sequential(
            normalization(channels),
            nn.SiLU(),
            conv_nd(dims, channels, self.out_channels, 3, padding=1),
        )

        self.up = up
        self.down = down
        self.updown = up or down

        if up:
            self.h_upd = Upsample(channels, False, dims)
            self.x_upd = Upsample(channels, False, dims)
        elif down:
            self.h_upd = Downsample(channels, False, dims)
            self.x_upd = Downsample(channels, False, dims)
        else:
            self.h_upd = self.x_upd = nn.Identity()

        self.emb_layers = nn.Sequential(
            nn.SiLU(),
            linear(
                emb_channels,
                2 * self.out_channels if use_scale_shift_norm else self.out_channels,
            ),
        )
        self.out_layers = nn.Sequential(
            normalization(self.out_channels),
            nn.SiLU(),
            nn.Dropout(p=dropout),
            zero_module(
                conv_nd(dims, self.out_channels, self.out_channels, 3, padding=1)
            ),
        )

        if self.out_channels == channels:
            self.skip_connection = nn.Identity()
        elif use_conv:
            self.skip_connection = conv_nd(
                dims, channels, self.out_channels, 3, padding=1
            )
        else:
            self.skip_connection = conv_nd(dims, channels, self.out_channels, 1)

    def forward(self, x, emb):
        """
        Apply the block to a Tensor, conditioned on a timestep embedding.
        :param x: an [N x C x ...] Tensor of features.
        :param emb: an [N x emb_channels] Tensor of timestep embeddings.
        :return: an [N x C x ...] Tensor of outputs.
        """
        return checkpoint(
            self._forward, (x, emb), self.parameters(), self.use_checkpoint
        )


    def _forward(self, x, emb):
        B, C, F, H, W = x.shape
        if self.dims == 2:
            x = rearrange(x, 'b c f h w -> (b f) c h w')
        elif self.dims == 1:
            x = rearrange(x, 'b c f h w -> (b h w) c f')
        else:
            raise NotImplementedError
        if self.updown:
            in_rest, in_conv = self.in_layers[:-1], self.in_layers[-1]
            h = in_rest(x)
            h = self.h_upd(h)
            x = self.x_upd(x)
            h = in_conv(h)
        else:
            h = self.in_layers(x)
        emb_out = self.emb_layers(emb).type(h.dtype)
        if self.dims == 2:
            emb_out = repeat(emb_out, 'b c -> (b f) c', f=F)
        elif self.dims == 1:
            emb_out = repeat(emb_out, 'b c -> (b h w) c', h=H, w=W)
        else:
            raise NotImplementedError
        while len(emb_out.shape) < len(h.shape):
            emb_out = emb_out[..., None]
        if self.use_scale_shift_norm:
            out_norm, out_rest = self.out_layers[0], self.out_layers[1:]
            scale, shift = th.chunk(emb_out, 2, dim=1)
            h = out_norm(h) * (1 + scale) + shift
            h = out_rest(h)
        else:
            h = h + emb_out
            h = self.out_layers(h)
        out = self.skip_connection(x) + h
        if self.dims == 2:
            out = rearrange(out, '(b f) c h w -> b c f h w', b=B, f=F)
        elif self.dims == 1:
            out = rearrange(out, '(b h w) c f -> b c f h w', b=B, h=H, w=W)
        else:
            raise NotImplementedError
        return out


class SptialTemporalResBlock(TimestepBlock):
    def __init__(self, SpatialResBlock : ResBlock):
        super().__init__()
        self.SpatialResBlock = SpatialResBlock
        self.TemporalResBlock = ResBlock(
            self.SpatialResBlock.out_channels,
            self.SpatialResBlock.emb_channels,
            self.SpatialResBlock.dropout,
            self.SpatialResBlock.out_channels,
            self.SpatialResBlock.use_conv,
            self.SpatialResBlock.use_scale_shift_norm,
            1,
            self.SpatialResBlock.use_checkpoint,
            False, False
        )

    def forward(self, x, emb):
        x = self.SpatialResBlock(x, emb)
        x = self.TemporalResBlock(x, emb)
        return x

class SptialTriplaneResBlock(TimestepBlock):
    def __init__(self, SpatialResBlock : ResBlock):
        super().__init__()
        self.SpatialResBlock = SpatialResBlock
        self.TriplaneResBlock = ResBlock(
            self.SpatialResBlock.out_channels,
            self.SpatialResBlock.emb_channels,
            self.SpatialResBlock.dropout,
            self.SpatialResBlock.out_channels,
            self.SpatialResBlock.use_conv,
            self.SpatialResBlock.use_scale_shift_norm,
            1,
            self.SpatialResBlock.use_checkpoint,
            False, False
        )

    def forward(self, x, emb):
        x = self.SpatialResBlock(x, emb)
        x = self.TriplaneResBlock(x, emb)
        return x

class NormResBlock(nn.Module):
    """
    A residual block that can optionally change the number of channels.
    :param channels: the number of input channels.
    :param emb_channels: the number of timestep embedding channels.
    :param dropout: the rate of dropout.
    :param out_channels: if specified, the number of out channels.
    :param use_conv: if True and out_channels is specified, use a spatial
        convolution instead of a smaller 1x1 convolution to change the
        channels in the skip connection.
    :param dims: determines if the signal is 1D, 2D, or 3D.
    :param use_checkpoint: if True, use gradient checkpointing on this module.
    :param up: if True, use this block for upsampling.
    :param down: if True, use this block for downsampling.
    """

    def __init__(
        self,
        channels,
        emb_channels,
        dropout,
        out_channels=None,
        use_conv=False,
        use_scale_shift_norm=False,
        dims=2,
        use_checkpoint=False,
        up=False,
        down=False,
        datasets=[],
    ):
        super().__init__()
        self.channels = channels
        self.emb_channels = emb_channels
        self.dropout = dropout
        self.out_channels = out_channels or channels
        self.use_conv = use_conv
        self.use_checkpoint = use_checkpoint
        self.use_scale_shift_norm = use_scale_shift_norm
        self.dims = dims

        self.in_layers = nn.Sequential(
            normalization(channels),
            nn.SiLU(),
            conv_nd(dims, channels, self.out_channels, 3, padding=1),
        )

        self.up = up
        self.down = down
        self.updown = up or down

        if up:
            self.h_upd = Upsample(channels, False, dims)
            self.x_upd = Upsample(channels, False, dims)
        elif down:
            self.h_upd = Downsample(channels, False, dims)
            self.x_upd = Downsample(channels, False, dims)
        else:
            self.h_upd = self.x_upd = nn.Identity()

        self.emb_layers = nn.Sequential(
            nn.SiLU(),
            linear(
                emb_channels,
                2 * self.out_channels if use_scale_shift_norm else self.out_channels,
            ),
        )
        self.out_layers = nn.Sequential(
            normalization(self.out_channels),
            nn.SiLU(),
            nn.Dropout(p=dropout),
            zero_module(
                conv_nd(dims, self.out_channels, self.out_channels, 3, padding=1)
            ),
        )

        if self.out_channels == channels:
            self.skip_connection = nn.Identity()
        elif use_conv:
            self.skip_connection = conv_nd(
                dims, channels, self.out_channels, 3, padding=1
            )
        else:
            self.skip_connection = conv_nd(dims, channels, self.out_channels, 1)

        self.datasets = datasets
        self.dataset_norm = nn.ParameterDict()
        for ds in datasets:
            self.dataset_norm[ds] = nn.Parameter(
                th.cat([th.ones(1, self.out_channels), th.zeros(1, self.out_channels)], 1)
            )

    def forward(self, x, emb, dataset):
        """
        Apply the block to a Tensor, conditioned on a timestep embedding.
        :param x: an [N x C x ...] Tensor of features.
        :param emb: an [N x emb_channels] Tensor of timestep embeddings.
        :return: an [N x C x ...] Tensor of outputs.
        """
        return checkpoint(
            self._forward, (x, emb, dataset), self.parameters(), self.use_checkpoint
        )


    def _forward(self, x, emb, dataset):
        B, C, F, H, W = x.shape
        if self.dims == 2:
            x = rearrange(x, 'b c f h w -> (b f) c h w')
        elif self.dims == 1:
            x = rearrange(x, 'b c f h w -> (b h w) c f')
        else:
            raise NotImplementedError
        if self.updown:
            in_rest, in_conv = self.in_layers[:-1], self.in_layers[-1]
            h = in_rest(x)
            h = self.h_upd(h)
            x = self.x_upd(x)
            h = in_conv(h)
        else:
            h = self.in_layers(x)

        emb_out = self.emb_layers(emb).type(h.dtype)
        if self.dims == 2:
            emb_out = repeat(emb_out, 'b c -> (b f) c', f=F)
        elif self.dims == 1:
            emb_out = repeat(emb_out, 'b c -> (b h w) c', h=H, w=W)
        else:
            raise NotImplementedError
        while len(emb_out.shape) < len(h.shape):
            emb_out = emb_out[..., None]

        if isinstance(dataset, str):
            dataset = B * [dataset]
        elif isinstance(dataset, list) or isinstance(dataset, tuple):
            assert len(dataset) == B
        else:
            raise NotImplementedError('Given dataset with type', type(dataset))
        dataset_out = []
        for ds in dataset:
            dataset_out.append(self.dataset_norm[ds])
        dataset_out = th.cat(dataset_out, 0)
        if self.dims == 2:
            dataset_out = repeat(dataset_out, 'b c -> (b f) c', f=F)
        elif self.dims == 1:
            dataset_out = repeat(dataset_out, 'b c -> (b h w) c', h=H, w=W)
        else:
            raise NotImplementedError
        while len(dataset_out.shape) < len(h.shape):
            dataset_out = dataset_out[..., None]

        out_norm, out_rest = self.out_layers[0], self.out_layers[1:]
        scale, shift = th.chunk(dataset_out, 2, dim=1)
        h = out_norm(h)
        h = h * scale + shift
        if self.use_scale_shift_norm:
            scale, shift = th.chunk(emb_out, 2, dim=1)
            h = h * scale + shift
        else:
            h = h + emb_out
        h = out_rest(h)

        out = self.skip_connection(x) + h
        if self.dims == 2:
            out = rearrange(out, '(b f) c h w -> b c f h w', b=B, f=F)
        elif self.dims == 1:
            out = rearrange(out, '(b h w) c f -> b c f h w', b=B, h=H, w=W)
        else:
            raise NotImplementedError
        return out


class NormResBlock_SpatialTransformer_and_UpDownsample(nn.Module):
    def __init__(
        self,
        UpDownModule,
        channels,
        emb_channels,
        dropout,
        out_channels=None,
        use_conv=False,
        use_scale_shift_norm=False,
        dims=2,
        use_checkpoint=False,
        up=False,
        down=False,
        num_heads=64,
        context_dim=None,
        use_linear_in_transformer=False,
        datasets=[],
        n_spatial_blocks=1,
    ):
        super().__init__()
        self.UpDownModule = UpDownModule
        if n_spatial_blocks > 1:
            self.NormResBlock = nn.ModuleList([
                NormResBlock(
                    channels,
                    emb_channels,
                    dropout,
                    out_channels,
                    use_conv,
                    use_scale_shift_norm,
                    dims,
                    use_checkpoint,
                    up,
                    down,
                    datasets,
                ) for _ in range(n_spatial_blocks)
            ])
        else:
            self.NormResBlock = NormResBlock(
                channels,
                emb_channels,
                dropout,
                out_channels,
                use_conv,
                use_scale_shift_norm,
                dims,
                use_checkpoint,
                up,
                down,
                datasets,
            )

        if num_heads == -1:
            d_head = 64
            num_heads = out_channels // d_head
        else:
            d_head = out_channels // num_heads
 
        if n_spatial_blocks > 1:
            self.spatialTransformer = nn.ModuleList([
                SpatialTransformer(
                    out_channels, 
                    num_heads, 
                    d_head, 
                    depth=1, 
                    context_dim=context_dim,
                    disable_self_attn=False, 
                    use_linear=use_linear_in_transformer,
                    use_checkpoint=use_checkpoint
                ) for _ in range(n_spatial_blocks)
            ])
        else:
            self.spatialTransformer = SpatialTransformer(
                out_channels, 
                num_heads, 
                d_head, 
                depth=1, 
                context_dim=context_dim,
                disable_self_attn=False, 
                use_linear=use_linear_in_transformer,
                use_checkpoint=use_checkpoint
            )

    def forward(self, x, emb, context, dataset):
        if isinstance(self.NormResBlock, NormResBlock):
            x = self.NormResBlock(x, emb, dataset)
        else:
            for m in self.NormResBlock:
                x = m(x, emb, dataset)
        if isinstance(self.spatialTransformer, SpatialTransformer):
            x = self.spatialTransformer(x, context)
        else:
            for m in self.spatialTransformer:
                x = m(x, context)
        return self.UpDownModule(x)


class SpatialTemporalNormResBlockTransformer_and_UpDownsample(nn.Module):
    def __init__(self, normResBlock_spatialTransformer_and_UpDownsample : NormResBlock_SpatialTransformer_and_UpDownsample):
        super().__init__()
        self.UpDownModule = normResBlock_spatialTransformer_and_UpDownsample.UpDownModule
        self.SpatialNormResBlock = normResBlock_spatialTransformer_and_UpDownsample.NormResBlock
        if isinstance(self.SpatialNormResBlock, NormResBlock):
            self.TemporalResBlock = ResBlock(
                self.SpatialNormResBlock.out_channels,
                self.SpatialNormResBlock.emb_channels,
                self.SpatialNormResBlock.dropout,
                self.SpatialNormResBlock.out_channels,
                self.SpatialNormResBlock.use_conv,
                self.SpatialNormResBlock.use_scale_shift_norm,
                1,
                self.SpatialNormResBlock.use_checkpoint,
                False, False
            )
        else:
            self.TemporalResBlock = nn.ModuleList([
                ResBlock(
                    m.out_channels,
                    m.emb_channels,
                    m.dropout,
                    m.out_channels,
                    m.use_conv,
                    m.use_scale_shift_norm,
                    1,
                    m.use_checkpoint,
                    False, False
                ) for m in self.SpatialNormResBlock
            ])
        self.spatialTransformer = normResBlock_spatialTransformer_and_UpDownsample.spatialTransformer
        if isinstance(self.spatialTransformer, SpatialTransformer):
            self.temporalTransformer = TemporalTransformer(
                self.spatialTransformer.in_channels,
                self.spatialTransformer.n_heads,
                self.spatialTransformer.d_head,
                self.spatialTransformer.depth,
                self.spatialTransformer.dropout,
                self.spatialTransformer.context_dim,
                self.spatialTransformer.disable_self_attn,
                self.spatialTransformer.use_linear,
                self.spatialTransformer.use_checkpoint
            )
        else:
            self.temporalTransformer = nn.ModuleList([
                TemporalTransformer(
                    m.in_channels,
                    m.n_heads,
                    m.d_head,
                    m.depth,
                    m.dropout,
                    m.context_dim,
                    m.disable_self_attn,
                    m.use_linear,
                    m.use_checkpoint
                ) for m in self.spatialTransformer
            ])

    def forward(self, x, emb, context, dataset):
        if isinstance(self.SpatialNormResBlock, NormResBlock):
            x = self.SpatialNormResBlock(x, emb, dataset)
            x = self.TemporalResBlock(x, emb)
        else:
            for m_spatial, m_temporal in zip(self.SpatialNormResBlock, self.TemporalResBlock):
                x = m_spatial(x, emb, dataset)
                x = m_temporal(x, emb)
        if isinstance(self.spatialTransformer, SpatialTransformer):
            x = self.spatialTransformer(x, context)
            x = self.temporalTransformer(x, context)
        else:
            for m_spatial, m_temporal in zip(self.spatialTransformer, self.temporalTransformer):
                x = m_spatial(x, context)
                x = m_temporal(x, context)
        return self.UpDownModule(x)

#### triplane
class SpatialTriplaneNormResBlockTransformer_and_UpDownsample(nn.Module):
    def __init__(self, normResBlock_spatialTransformer_and_UpDownsample : NormResBlock_SpatialTransformer_and_UpDownsample):
        super().__init__()
        self.UpDownModule = normResBlock_spatialTransformer_and_UpDownsample.UpDownModule
        self.SpatialNormResBlock = normResBlock_spatialTransformer_and_UpDownsample.NormResBlock
        if isinstance(self.SpatialNormResBlock, NormResBlock):
            self.TriplaneResBlock = ResBlock(
                self.SpatialNormResBlock.out_channels,
                self.SpatialNormResBlock.emb_channels,
                self.SpatialNormResBlock.dropout,
                self.SpatialNormResBlock.out_channels,
                self.SpatialNormResBlock.use_conv,
                self.SpatialNormResBlock.use_scale_shift_norm,
                1,
                self.SpatialNormResBlock.use_checkpoint,
                False, False
            )
        else:
            self.TriplaneResBlock = nn.ModuleList([
                ResBlock(
                    m.out_channels,
                    m.emb_channels,
                    m.dropout,
                    m.out_channels,
                    m.use_conv,
                    m.use_scale_shift_norm,
                    1,
                    m.use_checkpoint,
                    False, False
                ) for m in self.SpatialNormResBlock
            ])
        self.spatialTransformer = normResBlock_spatialTransformer_and_UpDownsample.spatialTransformer
        if isinstance(self.spatialTransformer, SpatialTransformer):
            self.triplaneTransformer = TriplaneTransformer(
                self.spatialTransformer.in_channels,
                self.spatialTransformer.n_heads,
                self.spatialTransformer.d_head,
                self.spatialTransformer.depth,
                self.spatialTransformer.dropout,
                self.spatialTransformer.context_dim,
                self.spatialTransformer.disable_self_attn,
                self.spatialTransformer.use_linear,
                self.spatialTransformer.use_checkpoint
            )
        else:
            self.triplaneTransformer = nn.ModuleList([
                TriplaneTransformer(
                    m.in_channels,
                    m.n_heads,
                    m.d_head,
                    m.depth,
                    m.dropout,
                    m.context_dim,
                    m.disable_self_attn,
                    m.use_linear,
                    m.use_checkpoint
                ) for m in self.spatialTransformer
            ])

    def forward(self, x, emb, context, dataset):
        if isinstance(self.SpatialNormResBlock, NormResBlock):
            x = self.SpatialNormResBlock(x, emb, dataset)
            x = self.TriplaneResBlock(x, emb)
        else:
            for m_spatial, m_triplane in zip(self.SpatialNormResBlock, self.TriplaneResBlock):
                x = m_spatial(x, emb, dataset)
                x = m_triplane(x, emb)
        if isinstance(self.spatialTransformer, SpatialTransformer):
            x = self.spatialTransformer(x, context)
            x = self.triplaneTransformer(x, context)
        else:
            for m_spatial, m_triplane in zip(self.spatialTransformer, self.triplaneTransformer):
                x = m_spatial(x, context)
                x = m_triplane(x, context)
        return self.UpDownModule(x)

class AttentionBlock(nn.Module):
    """
    An attention block that allows spatial positions to attend to each other.
    Originally ported from here, but adapted to the N-d case.
    https://github.com/hojonathanho/diffusion/blob/1e0dceb3b3495bbe19116a5e1b3596cd0706c543/diffusion_tf/models/unet.py#L66.
    """

    def __init__(
        self,
        channels,
        num_heads=1,
        num_head_channels=-1,
        use_checkpoint=False,
        use_new_attention_order=False,
    ):
        super().__init__()
        self.channels = channels
        if num_head_channels == -1:
            self.num_heads = num_heads
        else:
            assert (
                channels % num_head_channels == 0
            ), f"q,k,v channels {channels} is not divisible by num_head_channels {num_head_channels}"
            self.num_heads = channels // num_head_channels
        self.use_checkpoint = use_checkpoint
        self.norm = normalization(channels)
        self.qkv = conv_nd(1, channels, channels * 3, 1)
        if use_new_attention_order:
            # split qkv before split heads
            self.attention = QKVAttention(self.num_heads)
        else:
            # split heads before split qkv
            self.attention = QKVAttentionLegacy(self.num_heads)

        self.proj_out = zero_module(conv_nd(1, channels, channels, 1))

    def forward(self, x):
        return checkpoint(self._forward, (x,), self.parameters(), True)   # TODO: check checkpoint usage, is True # TODO: fix the .half call!!!
        #return pt_checkpoint(self._forward, x)  # pytorch

    def _forward(self, x):
        b, c, *spatial = x.shape
        x = x.reshape(b, c, -1)
        qkv = self.qkv(self.norm(x))
        h = self.attention(qkv)
        h = self.proj_out(h)
        return (x + h).reshape(b, c, *spatial)


class SpatialTemporalTransformer(nn.Module):
    def __init__(self, STransformer: SpatialTransformer):
        super().__init__()
        self.STransformer = STransformer
        self.TTransformer = TemporalTransformer(
            self.STransformer.in_channels,
            self.STransformer.n_heads,
            self.STransformer.d_head,
            self.STransformer.depth,
            self.STransformer.dropout,
            self.STransformer.context_dim,
            self.STransformer.disable_self_attn,
            self.STransformer.use_linear,
            self.STransformer.use_checkpoint
        )

    def forward(self, x, context=None):
        x = self.STransformer(x, context)
        x = self.TTransformer(x, context)
        return x

class SpatialTriplaneTransformer(nn.Module):
    def __init__(self, STransformer: SpatialTransformer):
        super().__init__()
        self.STransformer = STransformer
        self.TTransformer = TriplaneTransformer(
            self.STransformer.in_channels,
            self.STransformer.n_heads,
            self.STransformer.d_head,
            self.STransformer.depth,
            self.STransformer.dropout,
            self.STransformer.context_dim,
            self.STransformer.disable_self_attn,
            self.STransformer.use_linear,
            self.STransformer.use_checkpoint
        )

    def forward(self, x, context=None):
        x = self.STransformer(x, context)
        x = self.TTransformer(x, context)
        return x

def count_flops_attn(model, _x, y):
    """
    A counter for the `thop` package to count the operations in an
    attention operation.
    Meant to be used like:
        macs, params = thop.profile(
            model,
            inputs=(inputs, timestamps),
            custom_ops={QKVAttention: QKVAttention.count_flops},
        )
    """
    b, c, *spatial = y[0].shape
    num_spatial = int(np.prod(spatial))
    # We perform two matmuls with the same number of ops.
    # The first computes the weight matrix, the second computes
    # the combination of the value vectors.
    matmul_ops = 2 * b * (num_spatial ** 2) * c
    model.total_ops += th.DoubleTensor([matmul_ops])


class QKVAttentionLegacy(nn.Module):
    """
    A module which performs QKV attention. Matches legacy QKVAttention + input/ouput heads shaping
    """

    def __init__(self, n_heads):
        super().__init__()
        self.n_heads = n_heads

    def forward(self, qkv):
        """
        Apply QKV attention.
        :param qkv: an [N x (H * 3 * C) x T] tensor of Qs, Ks, and Vs.
        :return: an [N x (H * C) x T] tensor after attention.
        """
        bs, width, length = qkv.shape
        assert width % (3 * self.n_heads) == 0
        ch = width // (3 * self.n_heads)
        q, k, v = qkv.reshape(bs * self.n_heads, ch * 3, length).split(ch, dim=1)
        scale = 1 / math.sqrt(math.sqrt(ch))
        weight = th.einsum(
            "bct,bcs->bts", q * scale, k * scale
        )  # More stable with f16 than dividing afterwards
        weight = th.softmax(weight.float(), dim=-1).type(weight.dtype)
        a = th.einsum("bts,bcs->bct", weight, v)
        return a.reshape(bs, -1, length)

    @staticmethod
    def count_flops(model, _x, y):
        return count_flops_attn(model, _x, y)


class QKVAttention(nn.Module):
    """
    A module which performs QKV attention and splits in a different order.
    """

    def __init__(self, n_heads):
        super().__init__()
        self.n_heads = n_heads

    def forward(self, qkv):
        """
        Apply QKV attention.
        :param qkv: an [N x (3 * H * C) x T] tensor of Qs, Ks, and Vs.
        :return: an [N x (H * C) x T] tensor after attention.
        """
        bs, width, length = qkv.shape
        assert width % (3 * self.n_heads) == 0
        ch = width // (3 * self.n_heads)
        q, k, v = qkv.chunk(3, dim=1)
        scale = 1 / math.sqrt(math.sqrt(ch))
        weight = th.einsum(
            "bct,bcs->bts",
            (q * scale).view(bs * self.n_heads, ch, length),
            (k * scale).view(bs * self.n_heads, ch, length),
        )  # More stable with f16 than dividing afterwards
        weight = th.softmax(weight.float(), dim=-1).type(weight.dtype)
        a = th.einsum("bts,bcs->bct", weight, v.reshape(bs * self.n_heads, ch, length))
        return a.reshape(bs, -1, length)

    @staticmethod
    def count_flops(model, _x, y):
        return count_flops_attn(model, _x, y)


class Timestep(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, t):
        return timestep_embedding(t, self.dim)


class UNetModel(nn.Module):
    """
    The full UNet model with attention and timestep embedding.
    :param in_channels: channels in the input Tensor.
    :param model_channels: base channel count for the model.
    :param out_channels: channels in the output Tensor.
    :param num_res_blocks: number of residual blocks per downsample.
    :param attention_resolutions: a collection of downsample rates at which
        attention will take place. May be a set, list, or tuple.
        For example, if this contains 4, then at 4x downsampling, attention
        will be used.
    :param dropout: the dropout probability.
    :param channel_mult: channel multiplier for each level of the UNet.
    :param conv_resample: if True, use learned convolutions for upsampling and
        downsampling.
    :param dims: determines if the signal is 1D, 2D, or 3D.
    :param num_classes: if specified (as an int), then this model will be
        class-conditional with `num_classes` classes.
    :param use_checkpoint: use gradient checkpointing to reduce memory usage.
    :param num_heads: the number of attention heads in each attention layer.
    :param num_heads_channels: if specified, ignore num_heads and instead use
                               a fixed channel width per attention head.
    :param num_heads_upsample: works with num_heads to set a different number
                               of heads for upsampling. Deprecated.
    :param use_scale_shift_norm: use a FiLM-like conditioning mechanism.
    :param resblock_updown: use residual blocks for up/downsampling.
    :param use_new_attention_order: use a different attention pattern for potentially
                                    increased efficiency.
    """

    def __init__(
        self,
        image_size,
        in_channels,
        model_channels,
        out_channels,
        num_res_blocks,
        attention_resolutions,
        dropout=0,
        channel_mult=(1, 2, 4, 8),
        conv_resample=True,
        dims=2,
        num_classes=None,
        use_checkpoint=False,
        use_fp16=False,
        use_bf16=False,
        num_heads=-1,
        num_head_channels=-1,
        num_heads_upsample=-1,
        use_scale_shift_norm=False,
        resblock_updown=False,
        use_new_attention_order=False,
        use_spatial_transformer=False,    # custom transformer support
        transformer_depth=1,              # custom transformer support
        context_dim=None,                 # custom transformer support
        n_embed=None,                     # custom support for prediction of discrete ids into codebook of first stage vq model
        legacy=True,
        disable_self_attentions=None,
        num_attention_blocks=None,
        disable_middle_self_attn=False,
        use_linear_in_transformer=False,
        adm_in_channels=None,
        **extra_kwargs
    ):
        super().__init__()
        if use_spatial_transformer:
            assert context_dim is not None, 'Fool!! You forgot to include the dimension of your cross-attention conditioning...'

        self.context_dim = context_dim
        if context_dim is not None:
            assert use_spatial_transformer, 'Fool!! You forgot to use the spatial transformer for your cross-attention conditioning...'
            from omegaconf.listconfig import ListConfig
            if type(context_dim) == ListConfig:
                context_dim = list(context_dim)

        if num_heads_upsample == -1:
            num_heads_upsample = num_heads

        if num_heads == -1:
            assert num_head_channels != -1, 'Either num_heads or num_head_channels has to be set'

        if num_head_channels == -1:
            assert num_heads != -1, 'Either num_heads or num_head_channels has to be set'

        self.image_size = image_size
        self.in_channels = in_channels
        self.model_channels = model_channels
        self.out_channels = out_channels
        if isinstance(num_res_blocks, int):
            self.num_res_blocks = len(channel_mult) * [num_res_blocks]
        else:
            if len(num_res_blocks) != len(channel_mult):
                raise ValueError("provide num_res_blocks either as an int (globally constant) or "
                                 "as a list/tuple (per-level) with the same length as channel_mult")
            self.num_res_blocks = num_res_blocks
        if disable_self_attentions is not None:
            # should be a list of booleans, indicating whether to disable self-attention in TransformerBlocks or not
            assert len(disable_self_attentions) == len(channel_mult)
        if num_attention_blocks is not None:
            assert len(num_attention_blocks) == len(self.num_res_blocks)
            assert all(map(lambda i: self.num_res_blocks[i] >= num_attention_blocks[i], range(len(num_attention_blocks))))
            print(f"Constructor of UNetModel received num_attention_blocks={num_attention_blocks}. "
                  f"This option has LESS priority than attention_resolutions {attention_resolutions}, "
                  f"i.e., in cases where num_attention_blocks[i] > 0 but 2**i not in attention_resolutions, "
                  f"attention will still not be set.")

        self.attention_resolutions = attention_resolutions
        self.dropout = dropout
        self.channel_mult = channel_mult
        self.conv_resample = conv_resample
        self.num_classes = num_classes
        self.use_checkpoint = use_checkpoint
        self.dtype = th.float16 if use_fp16 else th.float32
        self.dtype = th.bfloat16 if use_bf16 else self.dtype
        self.num_heads = num_heads
        self.num_head_channels = num_head_channels
        self.num_heads_upsample = num_heads_upsample
        self.predict_codebook_ids = n_embed is not None
        self.dims = dims
        self.use_linear_in_transformer = use_linear_in_transformer
        self.spatial_finetune_blocks = extra_kwargs.get('spatial_finetune_blocks', 1)

        time_embed_dim = model_channels * 4
        self.time_embed = nn.Sequential(
            linear(model_channels, time_embed_dim),
            nn.SiLU(),
            linear(time_embed_dim, time_embed_dim),
        )
        self.time_embed_dim = time_embed_dim

        if self.num_classes is not None:
            if isinstance(self.num_classes, int):
                self.label_emb = nn.Embedding(num_classes, time_embed_dim)
            elif self.num_classes == "continuous":
                print("setting up linear c_adm embedding layer")
                self.label_emb = nn.Linear(1, time_embed_dim)
            elif self.num_classes == "sequential":
                assert adm_in_channels is not None
                self.label_emb = nn.Sequential(
                    nn.Sequential(
                        linear(adm_in_channels, time_embed_dim),
                        nn.SiLU(),
                        linear(time_embed_dim, time_embed_dim),
                    )
                )
            else:
                raise ValueError()

        self.input_blocks = nn.ModuleList(
            [
                TimestepEmbedSequential(
                    conv_nd(dims, in_channels, model_channels, 3, padding=1)
                )
            ]
        )
        self._feature_size = model_channels
        input_block_chans = [model_channels]
        ch = model_channels
        ds = 1
        for level, mult in enumerate(channel_mult):
            for nr in range(self.num_res_blocks[level]):
                layers = [
                    ResBlock(
                        ch,
                        time_embed_dim,
                        dropout,
                        out_channels=mult * model_channels,
                        dims=dims,
                        use_checkpoint=use_checkpoint,
                        use_scale_shift_norm=use_scale_shift_norm,
                    )
                ]
                ch = mult * model_channels
                if ds in attention_resolutions:
                    if num_head_channels == -1:
                        dim_head = ch // num_heads
                    else:
                        num_heads = ch // num_head_channels
                        dim_head = num_head_channels
                    if legacy:
                        #num_heads = 1
                        dim_head = ch // num_heads if use_spatial_transformer else num_head_channels
                    if exists(disable_self_attentions):
                        disabled_sa = disable_self_attentions[level]
                    else:
                        disabled_sa = False

                    if not exists(num_attention_blocks) or nr < num_attention_blocks[level]:
                        layers.append(
                            AttentionBlock(
                                ch,
                                use_checkpoint=use_checkpoint,
                                num_heads=num_heads,
                                num_head_channels=dim_head,
                                use_new_attention_order=use_new_attention_order,
                            ) if not use_spatial_transformer else SpatialTransformer(
                                ch, num_heads, dim_head, depth=transformer_depth, context_dim=context_dim,
                                disable_self_attn=disabled_sa, use_linear=use_linear_in_transformer,
                                use_checkpoint=use_checkpoint
                            )
                        )
                self.input_blocks.append(TimestepEmbedSequential(*layers))
                self._feature_size += ch
                input_block_chans.append(ch)
            if level != len(channel_mult) - 1:
                out_ch = ch
                self.input_blocks.append(
                    TimestepEmbedSequential(
                        ResBlock(
                            ch,
                            time_embed_dim,
                            dropout,
                            out_channels=out_ch,
                            dims=dims,
                            use_checkpoint=use_checkpoint,
                            use_scale_shift_norm=use_scale_shift_norm,
                            down=True,
                        )
                        if resblock_updown
                        else Downsample(
                            ch, conv_resample, dims=dims, out_channels=out_ch
                        )
                    )
                )
                ch = out_ch
                input_block_chans.append(ch)
                ds *= 2
                self._feature_size += ch

        if num_head_channels == -1:
            dim_head = ch // num_heads
        else:
            num_heads = ch // num_head_channels
            dim_head = num_head_channels
        if legacy:
            #num_heads = 1
            dim_head = ch // num_heads if use_spatial_transformer else num_head_channels

        layers = [
            ResBlock(
                ch,
                time_embed_dim,
                dropout,
                dims=dims,
                use_checkpoint=use_checkpoint,
                use_scale_shift_norm=use_scale_shift_norm,
            ),
        ]
        layers.append(
            AttentionBlock(
                ch,
                use_checkpoint=use_checkpoint,
                num_heads=num_heads,
                num_head_channels=dim_head,
                use_new_attention_order=use_new_attention_order,
            ) if not use_spatial_transformer else SpatialTransformer(  # always uses a self-attn
                            ch, num_heads, dim_head, depth=transformer_depth, context_dim=context_dim,
                            disable_self_attn=disable_middle_self_attn, use_linear=use_linear_in_transformer,
                            use_checkpoint=use_checkpoint
                        )
        )
        layers.append(
            ResBlock(
                ch,
                time_embed_dim,
                dropout,
                dims=dims,
                use_checkpoint=use_checkpoint,
                use_scale_shift_norm=use_scale_shift_norm,
            )
        )
        self.middle_block = TimestepEmbedSequential(*layers)
        self._feature_size += ch

        self.output_blocks = nn.ModuleList([])
        for level, mult in list(enumerate(channel_mult))[::-1]:
            for i in range(self.num_res_blocks[level] + 1):
                ich = input_block_chans.pop()
                layers = [
                    ResBlock(
                        ch + ich,
                        time_embed_dim,
                        dropout,
                        out_channels=model_channels * mult,
                        dims=dims,
                        use_checkpoint=use_checkpoint,
                        use_scale_shift_norm=use_scale_shift_norm,
                    )
                ]
                ch = model_channels * mult
                if ds in attention_resolutions:
                    if num_head_channels == -1:
                        dim_head = ch // num_heads
                    else:
                        num_heads = ch // num_head_channels
                        dim_head = num_head_channels
                    if legacy:
                        #num_heads = 1
                        dim_head = ch // num_heads if use_spatial_transformer else num_head_channels
                    if exists(disable_self_attentions):
                        disabled_sa = disable_self_attentions[level]
                    else:
                        disabled_sa = False

                    if not exists(num_attention_blocks) or i < num_attention_blocks[level]:
                        layers.append(
                            AttentionBlock(
                                ch,
                                use_checkpoint=use_checkpoint,
                                num_heads=num_heads_upsample,
                                num_head_channels=dim_head,
                                use_new_attention_order=use_new_attention_order,
                            ) if not use_spatial_transformer else SpatialTransformer(
                                ch, num_heads, dim_head, depth=transformer_depth, context_dim=context_dim,
                                disable_self_attn=disabled_sa, use_linear=use_linear_in_transformer,
                                use_checkpoint=use_checkpoint
                            )
                        )
                if level and i == self.num_res_blocks[level]:
                    out_ch = ch
                    layers.append(
                        ResBlock(
                            ch,
                            time_embed_dim,
                            dropout,
                            out_channels=out_ch,
                            dims=dims,
                            use_checkpoint=use_checkpoint,
                            use_scale_shift_norm=use_scale_shift_norm,
                            up=True,
                        )
                        if resblock_updown
                        else Upsample(ch, conv_resample, dims=dims, out_channels=out_ch)
                    )
                    ds //= 2
                self.output_blocks.append(TimestepEmbedSequential(*layers))
                self._feature_size += ch

        ## higher render channels
        self.out = nn.Sequential(
            normalization(ch),
            nn.SiLU(),
            zero_module(conv_nd(dims, model_channels, out_channels, 3, padding=1)),
            # zero_module(conv_nd(dims, model_channels, 80, 3, padding=1)),
        )

        if self.predict_codebook_ids:
            self.id_predictor = nn.Sequential(
            normalization(ch),
            conv_nd(dims, model_channels, n_embed, 1),
            #nn.LogSoftmax(dim=1)  # change to cross_entropy and produce non-normalized logits
        )

    def apply_extra_spatial(self, datasets):
        # print(self.input_blocks)
        for m in self.input_blocks:
            for i in range(len(m)):
                if isinstance(m[i], Downsample):
                    m[i] = NormResBlock_SpatialTransformer_and_UpDownsample(
                        m[i],
                        m[i].channels,
                        self.time_embed_dim,
                        self.dropout,
                        m[i].channels,
                        dims=self.dims,
                        use_checkpoint=self.use_checkpoint,
                        num_heads=self.num_heads,
                        context_dim=self.context_dim,
                        use_linear_in_transformer=self.use_linear_in_transformer,
                        datasets=datasets,
                        n_spatial_blocks=self.spatial_finetune_blocks
                    )
        #     print(m)   
        # print(self.input_blocks)
        # exit()
        for m in self.output_blocks:
            for i in range(len(m)):
                if isinstance(m[i], Upsample):
                    m[i] = NormResBlock_SpatialTransformer_and_UpDownsample(
                        m[i],
                        m[i].channels,
                        self.time_embed_dim,
                        self.dropout,
                        m[i].channels,
                        dims=self.dims,
                        use_checkpoint=self.use_checkpoint,
                        num_heads=self.num_heads,
                        context_dim=self.context_dim,
                        use_linear_in_transformer=self.use_linear_in_transformer,
                        datasets=datasets,
                        n_spatial_blocks=self.spatial_finetune_blocks
                    )
        # print(self.output_blocks)
        # exit()
    def apply_temporal(self):
        for m in self.input_blocks:
            for i in range(len(m)):
                if isinstance(m[i], ResBlock):
                    m[i] = SptialTemporalResBlock(m[i])
                elif isinstance(m[i], SpatialTransformer):
                    m[i] = SpatialTemporalTransformer(m[i])
                elif isinstance(m[i], NormResBlock_SpatialTransformer_and_UpDownsample):
                    m[i] = SpatialTemporalNormResBlockTransformer_and_UpDownsample(m[i])

        for i in range(len(self.middle_block)):
            if isinstance(self.middle_block[i], ResBlock):
                self.middle_block[i] = SptialTemporalResBlock(self.middle_block[i])
            elif isinstance(self.middle_block[i], SpatialTransformer):
                self.middle_block[i] = SpatialTemporalTransformer(self.middle_block[i])

        for m in self.output_blocks:
            for i in range(len(m)):
                if isinstance(m[i], ResBlock):
                    m[i] = SptialTemporalResBlock(m[i])
                elif isinstance(m[i], SpatialTransformer):
                    m[i] = SpatialTemporalTransformer(m[i])
                elif isinstance(m[i], NormResBlock_SpatialTransformer_and_UpDownsample):
                    m[i] = SpatialTemporalNormResBlockTransformer_and_UpDownsample(m[i])

    def apply_extra_triattn(self):
        for m in self.input_blocks:
            for i in range(len(m)):
                # print(i)
                if isinstance(m[i], ResBlock):
                    m[i] = SptialTriplaneResBlock(m[i])
                elif isinstance(m[i], SpatialTransformer):
                    m[i] = SpatialTriplaneTransformer(m[i])
                elif isinstance(m[i], NormResBlock_SpatialTransformer_and_UpDownsample):
                    m[i] = SpatialTriplaneNormResBlockTransformer_and_UpDownsample(m[i])
        # print(self.input_blocks)
        # exit()

        for i in range(len(self.middle_block)):
            if isinstance(self.middle_block[i], ResBlock):
                self.middle_block[i] = SptialTriplaneResBlock(self.middle_block[i])
            elif isinstance(self.middle_block[i], SpatialTransformer):
                self.middle_block[i] = SpatialTriplaneTransformer(self.middle_block[i])

        for m in self.output_blocks:
            for i in range(len(m)):
                if isinstance(m[i], ResBlock):
                    m[i] = SptialTriplaneResBlock(m[i])
                elif isinstance(m[i], SpatialTransformer):
                    m[i] = SpatialTriplaneTransformer(m[i])
                elif isinstance(m[i], NormResBlock_SpatialTransformer_and_UpDownsample):
                    m[i] = SpatialTriplaneNormResBlockTransformer_and_UpDownsample(m[i])
    
    def convert_to_fp16(self):
        """
        Convert the torso of the model to float16.
        """
        self.input_blocks.apply(convert_module_to_f16)
        self.middle_block.apply(convert_module_to_f16)
        self.output_blocks.apply(convert_module_to_f16)

    def convert_to_fp32(self):
        """
        Convert the torso of the model to float32.
        """
        self.input_blocks.apply(convert_module_to_f32)
        self.middle_block.apply(convert_module_to_f32)
        self.output_blocks.apply(convert_module_to_f32)

    def forward(self, x, timesteps=None, context=None, datasets=[], y=None, **kwargs):
        """
        Apply the model to an input batch.
        :param x: an [N x C x ...] Tensor of inputs.
        :param timesteps: a 1-D batch of timesteps.
        :param context: conditioning plugged in via crossattn
        :param y: an [N] Tensor of labels, if class-conditional.
        :return: an [N x C x ...] Tensor of outputs.
        """
        assert (y is not None) == (
            self.num_classes is not None
        ), "must specify y if and only if the model is class-conditional"
        hs = []
        t_emb = timestep_embedding(timesteps, self.model_channels, repeat_only=False)
        # print("dtpye", x.dtype)
        emb = self.time_embed(t_emb.to(x.dtype))
        #


        if self.num_classes is not None:
            assert y.shape[0] == x.shape[0]
            emb = emb + self.label_emb(y)

        h = x.type(self.dtype)

        initial_conv = self.input_blocks[0]

        f = h.shape[2]
        h = rearrange(h, 'b c f h w -> (b f) c h w')
        # print('h',h.shape)
        h = initial_conv(h, emb)
        # print('h_conv',h.shape)
        h = rearrange(h, '(b f) c h w -> b c f h w', f=f)
        hs.append(h)



        for module in self.input_blocks[1:]:
            h = module(h, emb, context, datasets)
            hs.append(h)



        h = self.middle_block(h, emb, context, datasets)


        for module in self.output_blocks:
            h = th.cat([h, hs.pop()], dim=1)
            h = module(h, emb, context, datasets)

        h = h.type(x.dtype)
        if self.predict_codebook_ids:
            return self.id_predictor(h)
        else:
            B, C, F, H, W = h.shape
            h = rearrange(h, 'b c f h w -> (b f) c h w')
            out = self.out(h)
            # for name, param in self.out[2].named_parameters():
                # print(name)
                # print("0-4",param)
            # print("4-",self.out[2])
            # print("0-4",self.out[2].weight[:4,:4])
            # print(self.out[2].weight[4:,:4])
            # # # print(self.out[2].weight)
            # exit()
            out = rearrange(out, '(b f) c h w -> b c f h w', b=B, f=F)
            return out

    
