# Copyright (c) OpenMMLab. All rights reserved.
from typing import Sequence

import numpy as np
import torch
import torch.nn as nn
from mmcv.cnn.bricks.transformer import FFN, PatchEmbed
from mmengine.model import BaseModule, ModuleList
from mmengine.model.weight_init import trunc_normal_

from mmpretrain.registry import MODELS
from mmpretrain.models.utils import (MultiheadAttention, SwiGLUFFNFused, build_norm_layer,
                     resize_pos_embed, to_2tuple)
from mmpretrain.models.backbones.base_backbone import BaseBackbone
from mamba_ssm import Mamba
from torch import Tensor
from typing import Optional
from mamba_ssm.ops.triton.layernorm import RMSNorm, layer_norm_fn, rms_norm_fn
from functools import partial
#from .ssm2d import Block2D,Mamba2D,SplitHead2D
from einops import rearrange
# from .mamband import MambaND
from mmpretrain.models.utils.embed import PatchMerging
from prettytable import PrettyTable

Block2D=SplitHead2D=nn.Identity # TODO: Clan implementation and release
MambaND = nn.Identity
class TransformerEncoderLayer(BaseModule):
    """Implements one encoder layer in Vision Transformer.

    Args:
        embed_dims (int): The feature dimension
        num_heads (int): Parallel attention heads
        feedforward_channels (int): The hidden dimension for FFNs
        layer_scale_init_value (float or torch.Tensor): Init value of layer
            scale. Defaults to 0.
        drop_rate (float): Probability of an element to be zeroed
            after the feed forward layer. Defaults to 0.
        attn_drop_rate (float): The drop out rate for attention output weights.
            Defaults to 0.
        drop_path_rate (float): Stochastic depth rate. Defaults to 0.
        num_fcs (int): The number of fully-connected layers for FFNs.
            Defaults to 2.
        qkv_bias (bool): enable bias for qkv if True. Defaults to True.
        ffn_type (str): Select the type of ffn layers. Defaults to 'origin'.
        act_cfg (dict): The activation config for FFNs.
            Defaults to ``dict(type='GELU')``.
        norm_cfg (dict): Config dict for normalization layer.
            Defaults to ``dict(type='LN')``.
        init_cfg (dict, optional): Initialization config dict.
            Defaults to None.
    """

    def __init__(self,
                 embed_dims,
                 num_heads,
                 feedforward_channels,
                 layer_scale_init_value=0.,
                 drop_rate=0.,
                 attn_drop_rate=0.,
                 drop_path_rate=0.,
                 num_fcs=2,
                 qkv_bias=True,
                 ffn_type='origin',
                 act_cfg=dict(type='GELU'),
                 norm_cfg=dict(type='LN'),
                 norm_cfg_2=dict(type='LN'),
                 init_cfg=None):
        super(TransformerEncoderLayer, self).__init__(init_cfg=init_cfg)

        self.embed_dims = embed_dims

        self.ln1 = build_norm_layer(norm_cfg, self.embed_dims)

        self.attn = MultiheadAttention(
            embed_dims=embed_dims,
            num_heads=num_heads,
            attn_drop=attn_drop_rate,
            proj_drop=drop_rate,
            dropout_layer=dict(type='DropPath', drop_prob=drop_path_rate),
            qkv_bias=qkv_bias,
            layer_scale_init_value=layer_scale_init_value)

        self.ln2 = build_norm_layer(norm_cfg_2 or norm_cfg, self.embed_dims)

        if ffn_type == 'origin':
            self.ffn = FFN(
                embed_dims=embed_dims,
                feedforward_channels=feedforward_channels,
                num_fcs=num_fcs,
                ffn_drop=drop_rate,
                dropout_layer=dict(type='DropPath', drop_prob=drop_path_rate),
                act_cfg=act_cfg,
                layer_scale_init_value=layer_scale_init_value)
        elif ffn_type == 'swiglu_fused':
            self.ffn = SwiGLUFFNFused(
                embed_dims=embed_dims,
                feedforward_channels=feedforward_channels,
                layer_scale_init_value=layer_scale_init_value)
        else:
            raise NotImplementedError

    @property
    def norm1(self):
        return self.ln1

    @property
    def norm2(self):
        return self.ln2

    def init_weights(self):
        super(TransformerEncoderLayer, self).init_weights()
        for m in self.ffn.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.normal_(m.bias, std=1e-6)

    def forward(self, x):
        x = x + self.attn(self.ln1(x))
        x = self.ffn(self.ln2(x), identity=x)
        return x
from mmcv.cnn.bricks.drop import build_dropout
class Block(nn.Module):
    def __init__(
        self, dim, mixer_cls, norm_cls=nn.LayerNorm, fused_add_norm=False, residual_in_fp32=False,reverse=False,
        transpose=False,split_head=False,
        drop_path_rate=0.0,drop_rate=0.0,use_mlp=False,downsample=False,
    ):
        """
        Simple block wrapping a mixer class with LayerNorm/RMSNorm and residual connection"

        This Block has a slightly different structure compared to a regular
        prenorm Transformer block.
        The standard block is: LN -> MHA/MLP -> Add.
        [Ref: https://arxiv.org/abs/2002.04745]
        Here we have: Add -> LN -> Mixer, returning both
        the hidden_states (output of the mixer) and the residual.
        This is purely for performance reasons, as we can fuse add and LayerNorm.
        The residual needs to be provided (except for the very first block).
        """
        super().__init__()
        self.residual_in_fp32 = residual_in_fp32
        self.fused_add_norm = fused_add_norm
        self.mixer = mixer_cls(dim)
        self.norm = norm_cls(dim)
        self.split_head = split_head
        self.reverse = reverse
        self.transpose = transpose
        self.drop_path = build_dropout(
            dict(type='DropPath', drop_prob=drop_path_rate)
        )
        self.dropout = build_dropout(
            dict(type='Dropout', drop_prob=drop_rate)
        )
        self.downsample = downsample
        if downsample:
            self.down_sample_layer = PatchMerging(
                dim,dim if self.split_head else dim,
            )
        if use_mlp:
            self.ffn = SwiGLUFFNFused(
                    embed_dims=dim,
                    feedforward_channels=int(dim*4),
                    layer_scale_init_value=0.0)
            self.ln2 = build_norm_layer(dict(type='LN'), dim)
        else:
            self.ffn = None
        if self.fused_add_norm:
            assert RMSNorm is not None, "RMSNorm import fails"
            assert isinstance(
                self.norm, (nn.LayerNorm, RMSNorm)
            ), "Only LayerNorm and RMSNorm are supported for fused_add_norm"

    def forward(
        self, hidden_states: Tensor, residual: Optional[Tensor] = None, inference_params=None,skip=True,**kwargs
    ):
        r"""Pass the input through the encoder layer.

        Args:
            hidden_states: the sequence to the encoder layer (required).
            residual: hidden_states = Mixer(LN(residual))
        """
        h = w = 0
        if self.transpose:
            l = hidden_states.shape[1]
            h = w = int(np.sqrt(l))
            # assert h * w == l
            hidden_states = rearrange(hidden_states,'n (h w) c -> n (w h) c',h=h,w=w)
            if residual is not None:
                residual = rearrange(residual,'n (h w) c -> n (w h) c',h=h,w=w)
        if self.reverse:
            hidden_states = hidden_states.flip(1)
            if residual is not None:
                residual = residual.flip(1)
        if not self.fused_add_norm:
            hidden_states = self.norm(hidden_states)
            # residual = (hidden_states + residual) if residual is not None else hidden_states
            # hidden_states = self.norm(residual.to(dtype=self.norm.weight.dtype))
            # if self.residual_in_fp32:
            #     residual = residual.to(torch.float32)
            if self.split_head:
                l = hidden_states.shape[1]
                h = w = int(np.sqrt(l))
                hidden_states = SplitHead2D.apply(hidden_states,4,h,w)
            if skip:
                hidden_states = hidden_states + self.drop_path(self.mixer(hidden_states, inference_params=inference_params,**(kwargs if isinstance(self.mixer,MambaND) else {})))
            else:
                hidden_states = self.drop_path(self.dropout(self.mixer(hidden_states, inference_params=inference_params)))
            if self.split_head:
                hidden_states = SplitHead2D.apply(hidden_states,4,h,w)
        else:
            fused_add_norm_fn = rms_norm_fn if isinstance(self.norm, RMSNorm) else layer_norm_fn
            hidden_states, residual = fused_add_norm_fn(
                hidden_states,
                self.norm.weight,
                self.norm.bias,
                residual=residual,
                prenorm=True,
                residual_in_fp32=self.residual_in_fp32,
                eps=self.norm.eps,
            )
            hidden_states = self.drop_path(self.mixer(hidden_states, inference_params=inference_params,**kwargs))
        if self.ffn is not None:
            hidden_states = self.ffn(self.ln2(hidden_states),identity=hidden_states)
        if self.reverse:
            hidden_states = hidden_states.flip(1)
            if residual is not None:
                residual = residual.flip(1)
        if self.transpose:
            hidden_states = rearrange(hidden_states,'n (w h) c -> n (h w) c',h=h,w=w)
            if residual is not None:
                residual = rearrange(residual,'n (w h) c -> n (h w) c',h=h,w=w)
        if self.downsample:
            if 'h' in kwargs:
                h,w = kwargs['h'],kwargs['w']
            hidden_states,(h,w) = self.down_sample_layer(
                hidden_states,(h,w)
            )
            assert residual is None
            residual = (h,w)
        return hidden_states, residual

    def allocate_inference_cache(self, batch_size, max_seqlen, dtype=None, **kwargs):
        return self.mixer.allocate_inference_cache(batch_size, max_seqlen, dtype=dtype, **kwargs)

import torch.nn.functional as F
from mamba_ssm.ops.selective_scan_interface import selective_scan_ref
def  causal_conv1d_fn_ref(x, weight, bias=None,seq_idx=None, activation=None):
    """
    x: (batch, dim, seqlen)
    weight: (dim, width)
    bias: (dim,)
    activation: either None or "silu" or "swish"

    out: (batch, dim, seqlen)
    """
    seqlen=x.shape[-1]
    dim,width = weight.shape
    assert activation =='silu'
    x = F.conv1d(x, weight.unsqueeze(1), bias, padding=width.item() - 1, groups=dim.item())[..., :seqlen]
    return F.silu(x)

def mamba_inner_ref(
    xz, conv1d_weight, conv1d_bias, x_proj_weight, delta_proj_weight,
    out_proj_weight, out_proj_bias,
    A, B=None, C=None, D=None, delta_bias=None, B_proj_bias=None,
    C_proj_bias=None, delta_softplus=True
):
    L = xz.shape[-1]
    delta_rank = delta_proj_weight.shape[1]
    d_state = A.shape[-1] * (1 if not A.is_complex() else 2)
    x, z = xz.chunk(2, dim=1)
    x = causal_conv1d_fn_ref(x, rearrange(conv1d_weight, "d 1 w -> d w"), bias=conv1d_bias,seq_idx=None, activation="silu")
    # We're being very careful here about the layout, to avoid extra transposes.
    # We want delta to have d as the slowest moving dimension
    # and L as the fastest moving dimension, since those are what the ssm_scan kernel expects.
    x_dbl = F.linear(rearrange(x, 'b d l -> (b l) d'), x_proj_weight)  # (bl d)
    delta = delta_proj_weight @ x_dbl[:, :delta_rank].t()
    delta = rearrange(delta, "d (b l) -> b d l", l=L)
    if B is None:  # variable B
        B = x_dbl[:, delta_rank:delta_rank + d_state]  # (bl d)
        if B_proj_bias is not None:
            B = B + B_proj_bias.to(dtype=B.dtype)
        if not A.is_complex():
            B = rearrange(B, "(b l) dstate -> b dstate l", l=L).contiguous()
        else:
            B = rearrange(B, "(b l) (dstate two) -> b dstate (l two)", l=L, two=2).contiguous()
    if C is None:  # variable B
        C = x_dbl[:, -d_state:]  # (bl d)
        if C_proj_bias is not None:
            C = C + C_proj_bias.to(dtype=C.dtype)
        if not A.is_complex():
            C = rearrange(C, "(b l) dstate -> b dstate l", l=L).contiguous()
        else:
            C = rearrange(C, "(b l) (dstate two) -> b dstate (l two)", l=L, two=2).contiguous()
    y = selective_scan_ref(x, delta, A, B, C, D, z=z, delta_bias=delta_bias, delta_softplus=True)
    return F.linear(rearrange(y, "b d l -> b l d"), out_proj_weight, out_proj_bias)

# for FLOPS calc only
class Mamba_Ref(Mamba):

    def forward(self, hidden_states, inference_params=None):
        """
        hidden_states: (B, L, D)
        Returns: same shape as hidden_states
        """
        batch, seqlen, dim = hidden_states.shape

        conv_state, ssm_state = None, None
        if inference_params is not None:
            conv_state, ssm_state = self._get_states_from_cache(inference_params, batch)
            if inference_params.seqlen_offset > 0:
                # The states are updated inplace
                out, _, _ = self.step(hidden_states, conv_state, ssm_state)
                return out

        # We do matmul and transpose BLH -> HBL at the same time
        xz = rearrange(
            self.in_proj.weight @ rearrange(hidden_states, "b l d -> d (b l)"),
            "d (b l) -> b d l",
            l=seqlen,
        )
        if self.in_proj.bias is not None:
            xz = xz + rearrange(self.in_proj.bias.to(dtype=xz.dtype), "d -> d 1")

        A = -torch.exp(self.A_log.float())  # (d_inner, d_state)
        # In the backward pass we write dx and dz next to each other to avoid torch.cat
        if self.use_fast_path and inference_params is None:  # Doesn't support outputting the states
            out = mamba_inner_ref(
                xz,
                self.conv1d.weight,
                self.conv1d.bias,
                self.x_proj.weight,
                self.dt_proj.weight,
                self.out_proj.weight,
                self.out_proj.bias,
                A,
                None,  # input-dependent B
                None,  # input-dependent C
                self.D.float(),
                delta_bias=self.dt_proj.bias.float(),
                delta_softplus=True,
            )
        else:
            x, z = xz.chunk(2, dim=1)
            # Compute short convolution
            if conv_state is not None:
                # If we just take x[:, :, -self.d_conv :], it will error if seqlen < self.d_conv
                # Instead F.pad will pad with zeros if seqlen < self.d_conv, and truncate otherwise.
                conv_state.copy_(F.pad(x, (self.d_conv - x.shape[-1], 0)))  # Update state (B D W)
            if causal_conv1d_fn is None:
                x = self.act(self.conv1d(x)[..., :seqlen])
            else:
                assert self.activation in ["silu", "swish"]
                x = causal_conv1d_fn(
                    x=x,
                    weight=rearrange(self.conv1d.weight, "d 1 w -> d w"),
                    bias=self.conv1d.bias,
                    activation=self.activation,
                )

            # We're careful here about the layout, to avoid extra transposes.
            # We want dt to have d as the slowest moving dimension
            # and L as the fastest moving dimension, since those are what the ssm_scan kernel expects.
            x_dbl = self.x_proj(rearrange(x, "b d l -> (b l) d"))  # (bl d)
            dt, B, C = torch.split(x_dbl, [self.dt_rank, self.d_state, self.d_state], dim=-1)
            dt = self.dt_proj.weight @ dt.t()
            dt = rearrange(dt, "d (b l) -> b d l", l=seqlen)
            B = rearrange(B, "(b l) dstate -> b dstate l", l=seqlen).contiguous()
            C = rearrange(C, "(b l) dstate -> b dstate l", l=seqlen).contiguous()
            assert self.activation in ["silu", "swish"]
            y = selective_scan_ref(
                x,
                dt,
                A,
                B,
                C,
                self.D.float(),
                z=z,
                delta_bias=self.dt_proj.bias.float(),
                delta_softplus=True,
                return_last_state=ssm_state is not None,
            )
            if ssm_state is not None:
                y, last_state = y
                ssm_state.copy_(last_state)
            y = rearrange(y, "b d l -> b l d")
            out = self.out_proj(y)
        return out


def create_block(
    d_model,
    ssm_cfg=None,
    norm_epsilon=1e-5,
    rms_norm=False,
    residual_in_fp32=False,
    fused_add_norm=False,
    layer_idx=None,
    device=None,
    dtype=None,
    reverse=None,
    is_2d=False,
    drop_rate=0.1,
    drop_path_rate=0.1,
    use_mlp=False,
    transpose=False,
    split_head=False,
    use_nd=False,
    downsample=False,
    use_ref=False,
    n_dim=2,
):
    if ssm_cfg is None:
        ssm_cfg = {}
    factory_kwargs = {"device": device, "dtype": dtype}
    
    if use_nd: 
        #ssm_cfg['d_state'] *= 4
        transpose = False
        reverse = False
        mixer_cls = partial(MambaND , layer_idx=layer_idx, n_dim=n_dim,**ssm_cfg, **factory_kwargs)
    elif use_ref:
        mixer_cls = partial(Mamba_Ref, layer_idx=layer_idx, **ssm_cfg, **factory_kwargs)
    else:
        mixer_cls = partial(Mamba2D if is_2d else Mamba, layer_idx=layer_idx, **ssm_cfg, **factory_kwargs)
    norm_cls = partial(
        nn.LayerNorm if not rms_norm else RMSNorm, eps=norm_epsilon, **factory_kwargs
    )
    if is_2d:
        block = Block2D(
            d_model,
            mixer_cls,
            norm_cls=norm_cls,
            fused_add_norm=fused_add_norm,
            residual_in_fp32=residual_in_fp32,
            reverse=reverse,
            drop_rate=drop_rate,
            transpose=transpose,
            drop_path_rate=drop_path_rate,
        )
    else:
        block = Block(
            d_model,
            mixer_cls,
            norm_cls=norm_cls,
            fused_add_norm=fused_add_norm,
            residual_in_fp32=residual_in_fp32,
            reverse=reverse,
            transpose=transpose,
            drop_rate=drop_rate,
            use_mlp=use_mlp,
            drop_path_rate=drop_path_rate,
            split_head=split_head,
            downsample=downsample,
        )
    block.layer_idx = layer_idx
    return block 
@MODELS.register_module()
class Mamba2DModel(BaseBackbone):
    """Vision Transformer.

    A PyTorch implement of : `An Image is Worth 16x16 Words: Transformers
    for Image Recognition at Scale <https://arxiv.org/abs/2010.11929>`_

    Args:
        arch (str | dict): Vision Transformer architecture. If use string,
            choose from 'small', 'base', 'large', 'deit-tiny', 'deit-small'
            and 'deit-base'. If use dict, it should have below keys:

            - **embed_dims** (int): The dimensions of embedding.
            - **num_layers** (int): The number of transformer encoder layers.
            - **num_heads** (int): The number of heads in attention modules.
            - **feedforward_channels** (int): The hidden dimensions in
              feedforward modules.

            Defaults to 'base'.
        img_size (int | tuple): The expected input image shape. Because we
            support dynamic input shape, just set the argument to the most
            common input image shape. Defaults to 224.
        patch_size (int | tuple): The patch size in patch embedding.
            Defaults to 16.
        in_channels (int): The num of input channels. Defaults to 3.
        out_indices (Sequence | int): Output from which stages.
            Defaults to -1, means the last stage.
        drop_rate (float): Probability of an element to be zeroed.
            Defaults to 0.
        drop_path_rate (float): stochastic depth rate. Defaults to 0.
        qkv_bias (bool): Whether to add bias for qkv in attention modules.
            Defaults to True.
        norm_cfg (dict): Config dict for normalization layer.
            Defaults to ``dict(type='LN')``.
        final_norm (bool): Whether to add a additional layer to normalize
            final feature map. Defaults to True.
        out_type (str): The type of output features. Please choose from

            - ``"cls_token"``: The class token tensor with shape (B, C).
            - ``"featmap"``: The feature map tensor from the patch tokens
              with shape (B, C, H, W).
            - ``"avg_featmap"``: The global averaged feature map tensor
              with shape (B, C).
            - ``"raw"``: The raw feature tensor includes patch tokens and
              class tokens with shape (B, L, C).

            Defaults to ``"cls_token"``.
        with_cls_token (bool): Whether concatenating class token into image
            tokens as transformer input. Defaults to True.
        frozen_stages (int): Stages to be frozen (stop grad and set eval mode).
            -1 means not freezing any parameters. Defaults to -1.
        interpolate_mode (str): Select the interpolate mode for position
            embeding vector resize. Defaults to "bicubic".
        layer_scale_init_value (float or torch.Tensor): Init value of layer
            scale. Defaults to 0.
        patch_cfg (dict): Configs of patch embeding. Defaults to an empty dict.
        layer_cfgs (Sequence | dict): Configs of each transformer layer in
            encoder. Defaults to an empty dict.
        init_cfg (dict, optional): Initialization config dict.
            Defaults to None.
    """
    arch_zoo = {
        **dict.fromkeys(
            ['small'], {
                'embed_dims': 384,
                'num_layers': 12,
                'num_heads': 6,
                'feedforward_channels': 384 * 4
            }),
        **dict.fromkeys(
            ['base'], {
                'embed_dims': 768,
                'num_layers': 12,
                'num_heads': 12,
                'feedforward_channels': 768 * 4
            }),
    }
    num_extra_tokens = 1  # class token
    OUT_TYPES = {'raw', 'cls_token', 'featmap', 'avg_featmap'}

    def __init__(self,
                 arch='base',
                 img_size=224,
                 patch_size=16,
                 in_channels=3,
                 out_indices=-1,
                 drop_rate=0.,
                 drop_path_rate=0.,
                 qkv_bias=True,
                 norm_cfg=dict(type='LN', eps=1e-6),
                 norm_cfg_2=dict(type='LN', eps=1e-6),
                 final_norm=True,
                 out_type='cls_token',
                 with_cls_token=True,
                 frozen_stages=-1,
                 interpolate_mode='bicubic',
                 layer_scale_init_value=0.,
                 patch_cfg=dict(),
                 layer_cfgs=dict(),
                 pre_norm=False,
                 init_cfg=None,
                 is_2d=True,
                 use_v2=False,
                 force_a2=False,
                 embed_dims=None,
                 has_transpose=True,
                 fused_add_norm=True,
                 use_mlp=False,
                 split_head=False,
                 use_nd=False,
                 downsample=None,
                 expand=None,
                 constant_dim=False,
                 has_reverse=True,
                 update_interval=None,
                 duplicate=None,
                 n_dim=2,
                 num_layers=None,
                 use_ref=False,
                 d_state=16):
        super(Mamba2DModel, self).__init__(init_cfg)

        if isinstance(arch, str):
            arch = arch.lower()
            assert arch in set(self.arch_zoo), \
                f'Arch {arch} is not in default archs {set(self.arch_zoo)}'
            self.arch_settings = self.arch_zoo[arch]
        else:
            essential_keys = {
                'embed_dims', 'num_layers', 'num_heads', 'feedforward_channels'
            }
            assert isinstance(arch, dict) and essential_keys <= set(arch), \
                f'Custom arch needs a dict with keys {essential_keys}'
            self.arch_settings = arch
        self.update_interval = update_interval
        self.embed_dims = self.arch_settings['embed_dims'] 
        if embed_dims is not None:
            self.embed_dims = embed_dims
        if num_layers is None:
            num_layers = self.arch_settings['num_layers']
        self.num_layers = num_layers * (2 if not use_mlp else 1)
        self.downsample = self.arch_settings.get('downsample',downsample)
        if self.downsample is None:
            self.downsample = []
        #self.downsample = list((x+1)*2-1 for x in self.downsample)
        self.img_size = to_2tuple(img_size)
        self.is_2d = is_2d
        self.use_nd=use_nd

        # Set patch embedding
        _patch_cfg = dict(
            in_channels=in_channels,
            input_size=img_size,
            embed_dims=self.embed_dims,
            conv_type='Conv2d',
            kernel_size=patch_size,
            stride=patch_size,
            bias=not pre_norm,  # disable bias if pre_norm is used(e.g., CLIP)
        )
        _patch_cfg.update(patch_cfg)
        self.patch_embed = PatchEmbed(**_patch_cfg)
        self.patch_resolution = self.patch_embed.init_out_size
        num_patches = self.patch_resolution[0] * self.patch_resolution[1]

        # Set out type
        if out_type not in self.OUT_TYPES:
            raise ValueError(f'Unsupported `out_type` {out_type}, please '
                             f'choose from {self.OUT_TYPES}')
        self.out_type = out_type

        # Set cls token
        self.with_cls_token = with_cls_token
        if with_cls_token:
            self.cls_token = nn.Parameter(torch.zeros(1, 1, self.embed_dims))
        elif out_type != 'cls_token':
            self.cls_token = None
            self.num_extra_tokens = 0
        else:
            raise ValueError(
                'with_cls_token must be True when `out_type="cls_token"`.')

        # Set position embedding
        self.interpolate_mode = interpolate_mode
        self.pos_embed = nn.Parameter(
            torch.zeros(1, num_patches + self.num_extra_tokens,
                        self.embed_dims))
        self._register_load_state_dict_pre_hook(self._prepare_pos_embed)

        self.drop_after_pos = nn.Dropout(p=drop_rate)

        if isinstance(out_indices, int):
            out_indices = [out_indices]
        assert isinstance(out_indices, Sequence), \
            f'"out_indices" must by a sequence or int, ' \
            f'get {type(out_indices)} instead.'
        for i, index in enumerate(out_indices):
            if index < 0:
                out_indices[i] = self.num_layers + index
            assert 0 <= out_indices[i] <= self.num_layers, \
                f'Invalid out_indices {index}'
        self.out_indices = out_indices

        # stochastic depth decay rule
        dpr = np.linspace(0, drop_path_rate, self.num_layers)

        self.layers = ModuleList()
        if isinstance(layer_cfgs, dict):
            layer_cfgs = [layer_cfgs] * self.num_layers
        ssm_cfg={"d_state":d_state}
        if expand is not None:
            ssm_cfg['expand'] = expand
        if duplicate:
            ssm_cfg['duplicate'] = duplicate
        if use_v2 and is_2d:
            ssm_cfg['use_v2'] = use_v2
        if force_a2:
            ssm_cfg['force_a2'] = force_a2
        dim = self.embed_dims
        for i in range(self.num_layers):
            #self.layers.append(TransformerEncoderLayer(**_layer_cfg))
            do_downsample = i in self.downsample
            self.layers.append(
                create_block(
                d_model=dim,
                ssm_cfg=ssm_cfg,
                fused_add_norm=fused_add_norm,
                residual_in_fp32=True,
                drop_rate=drop_rate,
                drop_path_rate=dpr[i],
                reverse= (not split_head ) and (i % 2) > 0 and has_reverse,
                transpose = (not split_head ) and has_transpose and ( i % 4) >=2,
                use_mlp=use_mlp,
                is_2d=is_2d,
                rms_norm=False,
                split_head=split_head,
                use_nd=use_nd,
                downsample=do_downsample,
                n_dim=n_dim,
                use_ref=use_ref
                )
            )
            if do_downsample and not constant_dim:
                dim *= 2
                
        self.frozen_stages = frozen_stages
        if pre_norm:
            self.pre_norm = build_norm_layer(norm_cfg, dim)
        else:
            self.pre_norm = nn.Identity()

        self.final_norm = final_norm
        if final_norm:
            self.ln1 = build_norm_layer(norm_cfg, dim)
        if self.out_type == 'avg_featmap':
            self.ln2 = build_norm_layer(norm_cfg_2, dim)

        # freeze stages only when self.frozen_stages > 0
        if self.frozen_stages > 0:
            self._freeze_stages()
        self.count_parameters()
    @property
    def norm1(self):
        return self.ln1

    @property
    def norm2(self):
        return self.ln2

    def init_weights(self):
        super(Mamba2DModel, self).init_weights()

        if not (isinstance(self.init_cfg, dict)
                and self.init_cfg['type'] == 'Pretrained'):
            if self.pos_embed is not None:
                trunc_normal_(self.pos_embed, std=0.02)
        else:
            ckpt = self.init_cfg['checkpoint']
            ckpt = torch.load(ckpt,map_location='cpu')['state_dict']
            ckpt = {k.replace('backbone.','',1):v for k,v in ckpt.items() if  k.startswith('backbone.') }
            curr_dict = self.state_dict()
            for k,v in ckpt.items():
                if k in curr_dict and (v_new:=curr_dict[k]).shape != v.shape:
                    if 'patch_embed' in k:
                        # n c H W
                        v_resized = torch.nn.functional.interpolate(v,v_new.shape[-2:])
                        assert v_resized.shape == v_new.shape
                        ckpt[k] = v_resized
                    elif 'pos_embed' in k:
                        b,old_len,dim = v.shape
                        old_d = int(np.sqrt(old_len))
                        new_len = v_new.shape[1]
                        new_d = int(np.sqrt(new_len))
                        v_resized = v.reshape(b,old_d,old_d,dim).permute(0,3,1,2)
                        v_resized = torch.nn.functional.interpolate(v_resized,(new_d,new_d)).flatten(2).permute(0,2,1)
                        assert v_resized.shape == v_new.shape
                        ckpt[k] = v_resized
                    else:
                        print(k,v_new.shape,v.shape)
            res = self.load_state_dict(ckpt,strict=False)
            print('----------init-------------------')
            print(res)

    def _prepare_pos_embed(self, state_dict, prefix, *args, **kwargs):
        name = prefix + 'pos_embed'
        if name not in state_dict.keys():
            return

        ckpt_pos_embed_shape = state_dict[name].shape
        if (not self.with_cls_token
                and ckpt_pos_embed_shape[1] == self.pos_embed.shape[1] + 1):
            # Remove cls token from state dict if it's not used.
            state_dict[name] = state_dict[name][:, 1:]
            ckpt_pos_embed_shape = state_dict[name].shape

        if self.pos_embed.shape != ckpt_pos_embed_shape:
            from mmengine.logging import MMLogger
            logger = MMLogger.get_current_instance()
            logger.info(
                f'Resize the pos_embed shape from {ckpt_pos_embed_shape} '
                f'to {self.pos_embed.shape}.')

            ckpt_pos_embed_shape = to_2tuple(
                int(np.sqrt(ckpt_pos_embed_shape[1] - self.num_extra_tokens)))
            pos_embed_shape = self.patch_embed.init_out_size

            state_dict[name] = resize_pos_embed(state_dict[name],
                                                ckpt_pos_embed_shape,
                                                pos_embed_shape,
                                                self.interpolate_mode,
                                                self.num_extra_tokens)

    @staticmethod
    def resize_pos_embed(*args, **kwargs):
        """Interface for backward-compatibility."""
        return resize_pos_embed(*args, **kwargs)

    def _freeze_stages(self):
        # freeze position embedding
        if self.pos_embed is not None:
            self.pos_embed.requires_grad = False
        # set dropout to eval model
        self.drop_after_pos.eval()
        # freeze patch embedding
        self.patch_embed.eval()
        for param in self.patch_embed.parameters():
            param.requires_grad = False
        # freeze pre-norm
        for param in self.pre_norm.parameters():
            param.requires_grad = False
        # freeze cls_token
        if self.cls_token is not None:
            self.cls_token.requires_grad = False
        # freeze layers
        for i in range(1, self.frozen_stages + 1):
            m = self.layers[i - 1]
            m.eval()
            for param in m.parameters():
                param.requires_grad = False
        # freeze the last layer norm
        if self.frozen_stages == len(self.layers):
            if self.final_norm:
                self.ln1.eval()
                for param in self.ln1.parameters():
                    param.requires_grad = False

            if self.out_type == 'avg_featmap':
                self.ln2.eval()
                for param in self.ln2.parameters():
                    param.requires_grad = False

    def forward(self, x):
        B = x.shape[0]
        x, patch_resolution = self.patch_embed(x)
        x = x + resize_pos_embed(
            self.pos_embed,
            self.patch_resolution,
            patch_resolution,
            mode=self.interpolate_mode,
            num_extra_tokens=self.num_extra_tokens)[:,self.num_extra_tokens:]
        if self.is_2d:
            assert self.cls_token is  None
            x = rearrange(x,'n (h w) c-> n c h w',h=patch_resolution[0],w=patch_resolution[1])
        if self.cls_token is not None:
            # stole cls_tokens impl from Phil Wang, thanks
            cls_token = self.cls_token.expand(B, -1, -1)
            x = torch.cat((x,cls_token), dim=1) # append last
        x = self.drop_after_pos(x)

        x = self.pre_norm(x)
        h = patch_resolution[0]
        w = patch_resolution[1]
        outs = []
        residual = None
        if self.update_interval:
            raw_x = x
            for i,layer in enumerate(self.layers):
                z = i // 2
                x_l,residual = layer(raw_x,residual,skip=False,h=h,w=w)
                if layer.downsample:
                        h,w = residual
                        residual = None
                x = x + x_l
                if (i+1) % self.update_interval == 0 or i == len(self.layers) - 1:
                    raw_x = x
                #x = raw_x
                if i == len(self.layers) - 1:
                    x = (x + residual) if residual is not None else x
                if i == len(self.layers) - 1 and self.final_norm:
                    x = self.ln1(x)

                if i in self.out_indices:
                    outs.append(self._format_output(x, patch_resolution))
        else:
            for i, layer in enumerate(self.layers):
                if self.use_nd:
                    x,residual = layer(x,residual,h=h,w=w)
                    if layer.downsample:
                        h,w = residual
                        residual = None
                else:
                    x,residual = layer(x,residual,h=h,w=w)
                    if layer.downsample:
                        h,w = residual
                        residual = None

                if i == len(self.layers) - 1:
                    x = (x + residual) if residual is not None else x
                if i == len(self.layers) - 1 and self.final_norm:
                    x = self.ln1(x)

                if i in self.out_indices:
                    outs.append(self._format_output(x, patch_resolution))

        return tuple(outs)

    def count_parameters(self,model=None):
        if model is None:
            model = self
        table = PrettyTable(["Modules", "Parameters"])
        total_params = 0
        for name, parameter in model.named_parameters():
            if not parameter.requires_grad:
                continue
            params = parameter.numel()
            table.add_row([name, params])
            total_params += params
        self.total_parms = total_params
        print(table)
        print(f"Total Trainable Params: {total_params}")
        return total_params
    
    def _format_output(self, x, hw):
        if self.out_type == 'raw':
            return x
        if self.out_type == 'cls_token':
            return x[:, -1]
        if not self.is_2d:
            patch_token = x[:, self.num_extra_tokens:]
        else:
            patch_token = x
        if self.out_type == 'featmap':
            B = x.size(0)
            # (B, N, C) -> (B, H, W, C) -> (B, C, H, W)
            if self.is_2d:
                return patch_token
            else:
                return patch_token.reshape(B, *hw, -1).permute(0, 3, 1, 2)
        if self.out_type == 'avg_featmap':
            if self.is_2d:
                return self.ln2(patch_token.flatten(2).mean(dim=-1))
            else:
                return self.ln2(patch_token.mean(dim=1))

    def get_layer_depth(self, param_name: str, prefix: str = ''):
        """Get the layer-wise depth of a parameter.

        Args:
            param_name (str): The name of the parameter.
            prefix (str): The prefix for the parameter.
                Defaults to an empty string.

        Returns:
            Tuple[int, int]: The layer-wise depth and the num of layers.

        Note:
            The first depth is the stem module (``layer_depth=0``), and the
            last depth is the subsequent module (``layer_depth=num_layers-1``)
        """
        num_layers = self.num_layers + 2

        if not param_name.startswith(prefix):
            # For subsequent module like head
            return num_layers - 1, num_layers

        param_name = param_name[len(prefix):]

        if param_name in ('cls_token', 'pos_embed'):
            layer_depth = 0
        elif param_name.startswith('patch_embed'):
            layer_depth = 0
        elif param_name.startswith('layers'):
            layer_id = int(param_name.split('.')[1])
            layer_depth = layer_id + 1
        else:
            layer_depth = num_layers - 1

        return layer_depth, num_layers
