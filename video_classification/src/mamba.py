# Copyright (c) OpenMMLab. All rights reserved.
from typing import Sequence

import numpy as np
import torch
import torch.nn as nn
from mmcv.cnn.bricks.transformer import FFN, PatchEmbed
from mmengine.model import BaseModule, ModuleList
from mmengine.model.weight_init import trunc_normal_
from mmcv.cnn import build_norm_layer
from mmengine.utils import to_2tuple
from typing import Tuple
MambaND = nn.Identity # TODO: cleanup and release
# from ..utils import (MultiheadAttention, SwiGLUFFNFused, build_norm_layer,
#                      resize_pos_embed, to_2tuple)
def resize_pos_embed(pos_embed: torch.Tensor,
                     src_shape: Tuple[int],
                     dst_shape: Tuple[int],
                     mode: str = 'trilinear',
                     num_extra_tokens: int = 1) -> torch.Tensor:
    """Resize pos_embed weights.

    Args:
        pos_embed (torch.Tensor): Position embedding weights with shape
            [1, L, C].
        src_shape (tuple): The resolution of downsampled origin training
            image, in format (T, H, W).
        dst_shape (tuple): The resolution of downsampled new training
            image, in format (T, H, W).
        mode (str): Algorithm used for upsampling. Choose one from 'nearest',
            'linear', 'bilinear', 'bicubic' and 'trilinear'.
            Defaults to 'trilinear'.
        num_extra_tokens (int): The number of extra tokens, such as cls_token.
            Defaults to 1.

    Returns:
        torch.Tensor: The resized pos_embed of shape [1, L_new, C]
    """
    if src_shape[0] == dst_shape[0] and src_shape[1] == dst_shape[1] \
            and src_shape[2] == dst_shape[2]:
        return pos_embed
    assert pos_embed.ndim == 3, 'shape of pos_embed must be [1, L, C]'
    _, L, C = pos_embed.shape
    src_t, src_h, src_w = src_shape
    assert L == src_t * src_h * src_w + num_extra_tokens, \
        f"The length of `pos_embed` ({L}) doesn't match the expected " \
        f'shape ({src_t}*{src_h}*{src_w}+{num_extra_tokens}).' \
        'Please check the `img_size` argument.'
    extra_tokens = pos_embed[:, :num_extra_tokens]

    src_weight = pos_embed[:, num_extra_tokens:]
    src_weight = src_weight.reshape(1, src_t, src_h, src_w,
                                    C).permute(0, 4, 1, 2, 3)

    dst_weight = F.interpolate(
        src_weight, size=dst_shape, align_corners=False, mode=mode)
    dst_weight = torch.flatten(dst_weight, 2).transpose(1, 2)

    return torch.cat((extra_tokens, dst_weight), dim=1)



# from .base_backbone import BaseBackbone
from mamba_ssm import Mamba
from torch import Tensor
from typing import Optional
from mamba_ssm.ops.triton.layernorm import RMSNorm, layer_norm_fn, rms_norm_fn
from functools import partial
#from .ssm2d import Block2D,Mamba2D,SplitHead2D
Block2D = Mamba2D = SplitHead2D = nn.Identity
from einops import rearrange
from mmengine.logging import MMLogger
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
        drop_path_rate=0.0,drop_rate=0.0,use_mlp=False,
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
        self, hidden_states: Tensor, residual: Optional[Tensor] = None, inference_params=None,order='t l h w',
        shape=None,skip=True,n_dim_pos=4
    ):
        r"""Pass the input through the encoder layer.

        Args:
            hidden_states: the sequence to the encoder layer (required).
            residual: hidden_states = Mixer(LN(residual))
        """
        h = w = 0
        assert shape is not None
        t,l,h,w = shape
        if n_dim_pos != 4:
            order = order.split(' ')
            assert len(order) == 4
            trunc_n = 4 - n_dim_pos
            tgt_order = f"(n {' '.join(order[:trunc_n])}) ({' '.join(order[trunc_n:])}) c"
        else:
            tgt_order = f'n ({order}) c'
        hidden_states =  rearrange(hidden_states,f'n (t l h w ) c -> {tgt_order}',t=t,l=l,h=h,w=w)
        # if self.transpose:
        #     l = hidden_states.shape[1]
        #     h = w = int(np.sqrt(l))
        #     # assert h * w == l
        #     hidden_states = rearrange(hidden_states,'n (h w) c -> n (w h) c',h=h,w=w)
        #     if residual is not None:
        #         residual = rearrange(residual,'n (h w) c -> n (w h) c',h=h,w=w)
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
                hidden_states = hidden_states + self.drop_path(self.dropout(self.mixer(hidden_states, inference_params=inference_params)))
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
            hidden_states = self.drop_path(self.mixer(hidden_states, inference_params=inference_params))
        if self.ffn is not None:
            hidden_states = self.ffn(self.ln2(hidden_states),identity=hidden_states)
        if self.reverse:
            hidden_states = hidden_states.flip(1)
            if residual is not None:
                residual = residual.flip(1)
        hidden_states =  rearrange(hidden_states,f'{tgt_order}->n (t l h w ) c ',t=t,l=l,h=h,w=w)
        # if self.transpose:
        #     hidden_states = rearrange(hidden_states,'n (w h) c -> n (h w) c',h=h,w=w)
        #     if residual is not None:
        #         residual = rearrange(residual,'n (w h) c -> n (h w) c',h=h,w=w)
        return hidden_states

    def allocate_inference_cache(self, batch_size, max_seqlen, dtype=None, **kwargs):
        return self.mixer.allocate_inference_cache(batch_size, max_seqlen, dtype=dtype, **kwargs)

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
):
    if ssm_cfg is None:
        ssm_cfg = {}
    factory_kwargs = {"device": device, "dtype": dtype}
    if use_nd:
        transpose = False
        reverse = False
        mixer_cls = partial(MambaND , layer_idx=layer_idx, n_dim=3,**ssm_cfg, **factory_kwargs)
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
        )
    block.layer_idx = layer_idx
    return block 

from mmaction.registry import MODELS
from mmaction.utils import ConfigType, OptConfigType
from mmengine.runner.checkpoint import _load_checkpoint
import re
from prettytable import PrettyTable
@MODELS.register_module()
class Mamba3DModel(BaseModule):
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
   }
    num_extra_tokens = 1  # class token
    OUT_TYPES = {'raw', 'cls_token', 'featmap', 'avg_featmap'}

    def __init__(self,
                 arch='base',
                 img_size=224,
                 patch_size=16,
                 patch_size_temporal=2,
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
                 num_frames: int = 16,
                 inflate_len=False,
                 is_1d=False,
                 is_2d=True,
                 use_v2=False,
                 force_a2=False,
                 has_transpose=True,
                 fused_add_norm=True,
                 use_mlp=False,
                 split_head=False,
                 pretrained=None,
                 pretrained2d=True,
                 dt_scale=0.0,
                 dt_scale_tmp=0.0,
                 use_nd=False,
                 force_2d=False,
                 update_interval=None,
                 copy_weight=False,
                 factorization=None,
                 inlfate_policy=None,
                 n_dim_pos=4,
                 d_state=16):
        super(Mamba3DModel, self).__init__(init_cfg)
        self.force_2d = force_2d
        self.use_nd = use_nd
        self.inlfate_policy = inlfate_policy
        self.pretrained2d = pretrained2d
        self.pretrained = pretrained
        self.n_dim_pos = n_dim_pos
        self.factorization = factorization
        self.inflate_len = inflate_len
        self.update_interval = update_interval
        self.copy_weight = copy_weight
        if pretrained:
            self.init_cfg = dict(type='Pretrained', checkpoint=pretrained)
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

        self.embed_dims = self.arch_settings['embed_dims']
        if self.inflate_len == 4:
            self.num_layers = self.arch_settings['num_layers'] * 4
        elif self.inflate_len:
            self.num_layers = self.arch_settings['num_layers'] * 3
        
        else:
            self.num_layers = self.arch_settings['num_layers'] * (2 if not use_mlp else 1)
        self.img_size = (num_frames,img_size,img_size)
        self.is_2d = is_2d

        # Set patch embedding
        _patch_cfg = dict(
            in_channels=in_channels,
            input_size=self.img_size,
            embed_dims=self.embed_dims,
            conv_type='Conv3d',
            kernel_size=(patch_size_temporal, patch_size, patch_size),
            stride=(patch_size_temporal, patch_size, patch_size),
            bias=not pre_norm,  # disable bias if pre_norm is used(e.g., CLIP)
            padding=(0, 0, 0),
            dilation=(1, 1, 1)
        )
            #         in_channels=in_channels,
            # embed_dims=embed_dims,
            # conv_type='Conv3d',
            # kernel_size=(tubelet_size, patch_size, patch_size),
            # stride=(tubelet_size, patch_size, patch_size),
            # padding=(0, 0, 0),
            # dilation=(1, 1, 1)
        _patch_cfg.update(patch_cfg)
        self.patch_embed = PatchEmbed(**_patch_cfg)
        self.patch_resolution = self.patch_embed.init_out_size
        self.patch_resolution = (self.patch_resolution[0],self.patch_resolution[1],self.patch_resolution[1])
        num_patches = self.patch_resolution[0] * self.patch_resolution[1]  * self.patch_resolution[1]
        self.is_1d = is_1d
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
        if use_v2 and is_2d:
            ssm_cfg['use_v2'] = use_v2
        if force_a2:
            ssm_cfg['force_a2'] = force_a2
        if dt_scale > 0:
            ssm_cfg['dt_scale'] = dt_scale
        if dt_scale_tmp > 0 and (i//2)%3==2:
            ssm_cfg['dt_scale'] = dt_scale
        for i in range(self.num_layers):
            _layer_cfg = dict(
                embed_dims=self.embed_dims,
                num_heads=self.arch_settings['num_heads'],
                feedforward_channels=self.
                arch_settings['feedforward_channels'],
                layer_scale_init_value=layer_scale_init_value,
                drop_rate=drop_rate,
                drop_path_rate=dpr[i],
                qkv_bias=qkv_bias,
                norm_cfg=norm_cfg)
            _layer_cfg.update(layer_cfgs[i])
            #self.layers.append(TransformerEncoderLayer(**_layer_cfg))
            self.layers.append(
                create_block(
                d_model=self.embed_dims,
                ssm_cfg=ssm_cfg,
                fused_add_norm=fused_add_norm,
                residual_in_fp32=True,
                drop_rate=drop_rate,
                drop_path_rate=dpr[i],
                reverse= (not split_head ) and (i % 2) > 0,
                transpose = (not split_head ) and has_transpose and ( i % 4) >=2,
                use_mlp=use_mlp,
                is_2d=is_2d,
                rms_norm=False,
                split_head=split_head,
                use_nd=self.use_nd
                )
            )
        self.frozen_stages = frozen_stages
        if pre_norm:
            self.pre_norm = build_norm_layer(norm_cfg, self.embed_dims)
        else:
            self.pre_norm = nn.Identity()

        self.final_norm = final_norm
        if self.out_type == 'avg_featmap':
            self.ln1 = nn.Identity()
            self.ln2 = build_norm_layer(norm_cfg_2, self.embed_dims)
        elif final_norm:
            self.ln1 = build_norm_layer(norm_cfg, self.embed_dims)
        # if self.out_type == 'avg_featmap':
        #     self.ln2 = build_norm_layer(norm_cfg_2, self.embed_dims)

        # freeze stages only when self.frozen_stages > 0
        if self.frozen_stages > 0:
            self._freeze_stages()

    @property
    def norm1(self):
        return self.ln1

    @property
    def norm2(self):
        return self.ln2

    def init_weights(self):

        if not (isinstance(self.init_cfg, dict)
                and self.init_cfg['type'] == 'Pretrained'):
            if self.pos_embed is not None:
                trunc_normal_(self.pos_embed, std=0.02)
        if self.pretrained2d:
            logger = MMLogger.get_current_instance()
            logger.info(f'load model from: {self.pretrained}')
            # Inflate 2D model into 3D model.
            self.inflate_weights(logger)
        else:
            if self.pretrained:
                self.init_cfg = dict(
                    type='Pretrained', checkpoint=self.pretrained)
            super().init_weights()

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
        b, _, _, h, w = x.shape
        # h //= self.patch_size
        # w //= self.patch_size
        
        x, patch_resolution = self.patch_embed(x)
        patch_resolution = (patch_resolution[0],patch_resolution[1],patch_resolution[1])

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

        outs = []
        residual = None
        orders = (
                't l h w',
                't l w h',
                'w h t l'
        )
        if self.is_1d:
            orders = (
                't l h w',
                't l h w',
                't l h w',
            )
        if self.force_2d:
            orders = (
                't l h w',
                't l w h',
                't l h w',
            )
            
        n_dim_pos = [self.n_dim_pos ] * 3

        if self.factorization is not None:
            if self.factorization == 'hw_t':
                n_dim_pos = (2,2,4)
            elif self.factorization == 'h_w_t':
                n_dim_pos = (1,1,2)
        shape = (patch_resolution[0],1,patch_resolution[1],patch_resolution[2])
        raw_x = 0
        if self.update_interval:
            raw_x = x
            for i,blk in enumerate(self.layers):
                z = i // 2
                d = z % len(orders)
                x = x + blk(raw_x,order=orders[d],shape=shape,skip=False,n_dim_pos=n_dim_pos[d])
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
            for i,blk in enumerate(self.layers):
                z = i // 2
                d = z % len(orders)
                
                x = blk(x,order=orders[d],shape=shape,n_dim_pos=n_dim_pos[d])
                    
            # for i, layer in enumerate(self.layers):
            #     x,residual = layer(x,residual)

                if i == len(self.layers) - 1:
                    x = (x + residual) if residual is not None else x
                if i == len(self.layers) - 1 and self.final_norm:
                    x = self.ln1(x)

                if i in self.out_indices:
                    outs.append(self._format_output(x, patch_resolution))
        return outs[-1]
    
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

    def inflate_weights(self, logger: MMLogger) -> None:
        """Inflate the swin2d parameters to swin3d.

        The differences between swin3d and swin2d mainly lie in an extra
        axis. To utilize the pretrained parameters in 2d model, the weight
        of swin2d models should be inflated to fit in the shapes of the
        3d counterpart.

        Args:
            logger (MMLogger): The logger used to print debugging information.
        """
        if not self.pretrained:
            return
        checkpoint = _load_checkpoint(self.pretrained, map_location='cpu')
        state_dict = checkpoint['state_dict']
        state_dict = {k.replace('backbone.',''):v for k,v in state_dict.items()}
        curr_state_dict = self.state_dict()
        if self.inflate_len:
            new_weights = {}
            for k,v in state_dict.items():
                if 'layer' in k:
                    i_layer = int(re.compile('layers.([0-9]+).').findall(k)[0])
                    # 0 1 2 3 x x 4 5 6 7 x x
                    n_blk = i_layer // 4
                    n_idx = i_layer % 4
                    new_idx = n_blk * 6 + n_idx
                    k1 = k.replace(f'layers.{i_layer}',f'layers.{new_idx}')
                    assert k1 not in new_weights
                    new_weights[k1] = v
                    if self.copy_weight:
                        if  n_idx in [2,3]:
                            k2 = k.replace(f'layers.{i_layer}',f'layers.{new_idx+2}')
                            new_weights[k2] = v
                else:
                    new_weights[k] = v
            state_dict = new_weights
        for k in curr_state_dict:
            if k in state_dict:
                if (shape_1:=curr_state_dict[k].shape) != (shape_2:=state_dict[k].shape):
                    if 'patch_embed' in k:
                        state_dict[k] = state_dict[k].unsqueeze(-3).repeat(1,1,shape_1[2],1,1) / shape_1[2]
                        assert state_dict[k].shape ==shape_1
                    elif 'pos_embed' in k:
                        old_len = state_dict[k].shape[1]
                        state_dict[k] = state_dict[k].repeat(1,self.patch_resolution[0],1) #/ self.patch_resolution[0]
                        idxes = torch.arange(self.patch_resolution[0]).view(1,-1,1).repeat(1,1,old_len).view(1,-1,1).float()
                        if self.inlfate_policy == 'cosine':
                            state_dict[k]  = state_dict[k]  * torch.cos(idxes / self.patch_resolution[0] * np.pi)
                        elif self.inlfate_policy == 'single':
                            state_dict[k]  = state_dict[k]  * (idxes ==( self.patch_resolution[0]//2))
                        assert state_dict[k].shape ==shape_1,(state_dict[k].shape,shape_1)
                    else:
                        print(k,shape_1,shape_2)
            else:
                print(k)
                #re.compile('')
        # delete relative_position_index since we always re-init it
        
        # bicubic interpolate relative_position_bias_table if not match
        # relative_position_bias_table_keys = [
        #     k for k in state_dict.keys() if 'relative_position_bias_table' in k
        # ]
        # for k in relative_position_bias_table_keys:
        #     relative_position_bias_table_pretrained = state_dict[k]
        #     relative_position_bias_table_current = self.state_dict()[k]
        #     L1, nH1 = relative_position_bias_table_pretrained.size()
        #     L2, nH2 = relative_position_bias_table_current.size()
        #     L2 = (2 * self.window_size[1] - 1) * (2 * self.window_size[2] - 1)
        #     wd = self.window_size[0]
        #     if nH1 != nH2:
        #         logger.warning(f'Error in loading {k}, passing')
        #     else:
        #         if L1 != L2:
        #             S1 = int(L1**0.5)
        #             relative_position_bias_table_pretrained_resized = \
        #                 torch.nn.functional.interpolate(
        #                     relative_position_bias_table_pretrained.permute(
        #                         1, 0).view(1, nH1, S1, S1),
        #                     size=(2 * self.window_size[1] - 1,
        #                           2 * self.window_size[2] - 1),
        #                     mode='bicubic')
        #             relative_position_bias_table_pretrained = \
        #                 relative_position_bias_table_pretrained_resized. \
        #                 view(nH2, L2).permute(1, 0)
        #     state_dict[k] = relative_position_bias_table_pretrained.repeat(
        #         2 * wd - 1, 1)

        # In the original swin2d checkpoint, the last layer of the
        # backbone is the norm layer, and the original attribute
        # name is `norm`. We changed it to `norm3` which means it
        # is the last norm layer of stage 4.
        # if hasattr(self, 'norm3'):
        #     state_dict['norm3.weight'] = state_dict['norm.weight']
        #     state_dict['norm3.bias'] = state_dict['norm.bias']
        #     del state_dict['norm.weight']
        #     del state_dict['norm.bias']

        msg = self.load_state_dict(state_dict, strict=False)
        logger.info(msg)

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
                return patch_token.reshape(B, *hw, -1).permute(0, 4, 1, 2,3)
        if self.out_type == 'avg_featmap':
            if self.is_2d:
                return self.ln2(patch_token.mean(dim=1))
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
