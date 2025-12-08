# Copyright (c) MONAI Consortium
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#     http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import torch
import torch.nn as nn
from typing import Sequence
from model import Ensemble_Seg_Model, Ensemble_Context_Model, _get_association
from monai.utils import ensure_tuple_rep
from monai.networks.blocks import UnetOutBlock
from model.encoder.sp_swin import SP_SwinTransformer
from model.decoder.unetr_decoder_sp_simple import Unetr_decoder_SP_simple


class Salt_ensemble_simple(Ensemble_Seg_Model):
    """
    Swin UNETR based on: "Hatamizadeh et al.,
    Swin UNETR: Swin Transformers for Semantic Segmentation of Brain Tumors in MRI Images
    <https://arxiv.org/abs/2201.01266>"
    """

    def __init__(
        self,
        args,
        input_channel=3,
        criteria=None,
        img_size: int = 224,
        in_channels: int = 3,
        out_channels: int = 2,
        patch_res: int = 2,
        window_res: int = 7,
        depths: Sequence[int] = (2, 2, 2, 2),
        num_heads: Sequence[int] = (4, 8, 16, 32),
        feature_size: int = 24,  # embed dim
        space_embed_dim: int = 1024,
        norm_name: str = "instance",
        drop_rate: float = 0.0,
        attn_drop_rate: float = 0.0,
        dropout_path_rate: float = 0.0,
        normalize: bool = True,
        use_checkpoint: bool = False,
        spatial_dims: int = 2,
        downsample="sp_merging",
    ) -> None:
        """
        Args:
            img_size: dimension of input image.
            in_channels: dimension of input channels.
            out_channels: dimension of output channels.
            feature_size: dimension of network feature size.
            depths: number of layers in each stage.
            num_heads: number of attention heads.
            norm_name: feature normalization type and arguments.
            drop_rate: dropout rate.
            attn_drop_rate: attention dropout rate.
            dropout_path_rate: drop path rate.
            normalize: normalize output intermediate features in each stage.
            use_checkpoint: use gradient checkpointing for reduced memory usage.
            spatial_dims: number of spatial dims.
            downsample: module used for downsampling, available options are `"mergingv2"`, `"merging"` and a
                user-specified `nn.Module` following the API defined in :py:class:`monai.networks.nets.PatchMerging`.
                The default is currently `"merging"` (the original version defined in v0.9.0).

        Examples::

            # for 3D single channel input with size (96,96,96), 4-channel output and feature size of 48.
            >>> net = SwinUNETR(img_size=(96,96,96), in_channels=1, out_channels=4, feature_size=48)

            # for 3D 4-channel input with size (128,128,128), 3-channel output and (2,4,2,2) layers in each stage.
            >>> net = SwinUNETR(img_size=(128,128,128), in_channels=4, out_channels=3, depths=(2,4,2,2))

            # for 2D single channel input with size (96,96), 2-channel output and gradient checkpointing.
            >>> net = SwinUNETR(img_size=(96,96), in_channels=3, out_channels=2, use_checkpoint=True, spatial_dims=2)

        """

        super(Salt_ensemble_simple, self).__init__(args=args, criteria=criteria)

        self.patch_num = img_size // patch_res
        self.feature_size = feature_size

        img_size = ensure_tuple_rep(img_size, spatial_dims)
        patch_size = ensure_tuple_rep(patch_res, spatial_dims)
        window_size = ensure_tuple_rep(window_res, spatial_dims)

        if spatial_dims not in (2, 3):
            raise ValueError("spatial dimension should be 2 or 3.")

        if not (0 <= drop_rate <= 1):
            raise ValueError("dropout rate should be between 0 and 1.")

        if not (0 <= attn_drop_rate <= 1):
            raise ValueError("attention dropout rate should be between 0 and 1.")

        if not (0 <= dropout_path_rate <= 1):
            raise ValueError("drop path rate should be between 0 and 1.")

        # if feature_size % 12 != 0:
        #     raise ValueError("feature_size should be divisible by 12.")

        self.normalize = normalize

        self.encoder = SP_SwinTransformer(
            in_chans=in_channels,
            embed_dim=feature_size,
            window_size=window_size,
            patch_size=patch_size,
            depths=depths,
            num_heads=num_heads,
            space_embed_dim=space_embed_dim,
            mlp_ratio=4.0,
            qkv_bias=True,
            drop_rate=drop_rate,
            attn_drop_rate=attn_drop_rate,
            drop_path_rate=dropout_path_rate,
            norm_layer=nn.LayerNorm,
            use_checkpoint=use_checkpoint,
            spatial_dims=spatial_dims,
            downsample=downsample,
        )

        self.decoder = Unetr_decoder_SP_simple(
            in_channels=in_channels,
            out_channels=out_channels,
            patch_res=patch_res,
            feature_size=feature_size,
            norm_name=norm_name,
            normalize=normalize,
            spatial_dims=spatial_dims
        )

        self.msp = nn.Parameter(torch.zeros((self.args.batch_size, self.args.size ** 2, self.args.sp_num), dtype=torch.float32), requires_grad=False)
        self.ma2 = nn.Parameter(torch.zeros((self.args.batch_size, self.args.sp_num // 4, self.args.sp_num // 16), dtype=torch.float32), requires_grad=False)
        self.ma3 = nn.Parameter(torch.zeros((self.args.batch_size, self.args.sp_num // 16, self.args.sp_num // 64), dtype=torch.float32), requires_grad=False)
        self.ma4 = nn.Parameter(torch.zeros((self.args.batch_size, self.args.sp_num // 64, self.args.sp_num // 256), dtype=torch.float32), requires_grad=False)

        self.mspt = nn.Parameter(torch.zeros((self.args.test_batch_size, self.args.size ** 2, self.args.sp_num), dtype=torch.float32), requires_grad=False)
        self.ma2t = nn.Parameter(torch.zeros((self.args.test_batch_size, self.args.sp_num // 4, self.args.sp_num // 16), dtype=torch.float32), requires_grad=False)
        self.ma3t = nn.Parameter(torch.zeros((self.args.test_batch_size, self.args.sp_num // 16, self.args.sp_num // 64), dtype=torch.float32), requires_grad=False)
        self.ma4t = nn.Parameter(torch.zeros((self.args.test_batch_size, self.args.sp_num // 64, self.args.sp_num // 256), dtype=torch.float32), requires_grad=False)

    def forward(self, x_in, matrix_sp, a1, a2, a3, a4, inference=False):

        if inference:
            self.mspt, self.ma2t, self.ma3t, self.ma4t = _get_association(
                matrix_sp, a2, a3, a4,
                self.mspt, self.ma2t, self.ma3t, self.ma4t,
                self.args.size, self.args.sp_num, self.args.test_batch_size
            )

            hidden_states_out = self.encoder(x_in, self.mspt, a1, self.ma2t, self.ma3t, self.ma4t, self.normalize)
            out = self.decoder(x_in, hidden_states_out, self.mspt, a1, self.ma2t, self.ma3t, self.ma4t)
        else:
            self.msp, self.ma2, self.ma3, self.ma4 = _get_association(
                matrix_sp, a2, a3, a4,
                self.msp, self.ma2, self.ma3, self.ma4,
                self.args.size, self.args.sp_num, self.args.batch_size
            )

            hidden_states_out = self.encoder(x_in, self.msp, a1, self.ma2, self.ma3, self.ma4, self.normalize)
            out = self.decoder(x_in, hidden_states_out, self.msp, a1, self.ma2, self.ma3, self.ma4)

        return out

    def on_save_checkpoint(self, checkpoint):
        # pop the backbone here using custom logic
        del checkpoint['state_dict']['msp']
        del checkpoint['state_dict']['ma2']
        del checkpoint['state_dict']['ma3']
        del checkpoint['state_dict']['ma4']
        del checkpoint['state_dict']['mspt']
        del checkpoint['state_dict']['ma2t']
        del checkpoint['state_dict']['ma3t']
        del checkpoint['state_dict']['ma4t']

    def configure_optimizers(self):
        opt = torch.optim.Adam(self.parameters(), lr=self.args.lr)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=self.args.epoch)

        return [opt], [scheduler]


class Salt_ensemble_context(Ensemble_Context_Model):
    """
    Swin UNETR based on: "Hatamizadeh et al.,
    Swin UNETR: Swin Transformers for Semantic Segmentation of Brain Tumors in MRI Images
    <https://arxiv.org/abs/2201.01266>"
    """

    def __init__(
        self,
        args,
        input_channel=3,
        criteria=None,
        img_size: int = 224,
        in_channels: int = 3,
        out_channels: int = 3,
        patch_res: int = 2,
        window_res: int = 7,
        depths: Sequence[int] = (2, 2, 2, 2),
        num_heads: Sequence[int] = (4, 8, 16, 32),
        feature_size: int = 24,  # embed dim
        space_embed_dim: int = 1024,
        norm_name: str = "instance",
        drop_rate: float = 0.0,
        attn_drop_rate: float = 0.0,
        dropout_path_rate: float = 0.0,
        normalize: bool = True,
        use_checkpoint: bool = False,
        spatial_dims: int = 2,
        downsample="sp_merging",
    ) -> None:
        """
        Args:
            img_size: dimension of input image.
            in_channels: dimension of input channels.
            out_channels: dimension of output channels.
            feature_size: dimension of network feature size.
            depths: number of layers in each stage.
            num_heads: number of attention heads.
            norm_name: feature normalization type and arguments.
            drop_rate: dropout rate.
            attn_drop_rate: attention dropout rate.
            dropout_path_rate: drop path rate.
            normalize: normalize output intermediate features in each stage.
            use_checkpoint: use gradient checkpointing for reduced memory usage.
            spatial_dims: number of spatial dims.
            downsample: module used for downsampling, available options are `"mergingv2"`, `"merging"` and a
                user-specified `nn.Module` following the API defined in :py:class:`monai.networks.nets.PatchMerging`.
                The default is currently `"merging"` (the original version defined in v0.9.0).

        Examples::

            # for 3D single channel input with size (96,96,96), 4-channel output and feature size of 48.
            >>> net = SwinUNETR(img_size=(96,96,96), in_channels=1, out_channels=4, feature_size=48)

            # for 3D 4-channel input with size (128,128,128), 3-channel output and (2,4,2,2) layers in each stage.
            >>> net = SwinUNETR(img_size=(128,128,128), in_channels=4, out_channels=3, depths=(2,4,2,2))

            # for 2D single channel input with size (96,96), 2-channel output and gradient checkpointing.
            >>> net = SwinUNETR(img_size=(96,96), in_channels=3, out_channels=2, use_checkpoint=True, spatial_dims=2)

        """

        super(Salt_ensemble_context, self).__init__(args=args, criteria=criteria)

        self.patch_num = img_size // patch_res
        self.feature_size = feature_size

        img_size = ensure_tuple_rep(img_size, spatial_dims)
        patch_size = ensure_tuple_rep(patch_res, spatial_dims)
        window_size = ensure_tuple_rep(window_res, spatial_dims)

        if spatial_dims not in (2, 3):
            raise ValueError("spatial dimension should be 2 or 3.")

        if not (0 <= drop_rate <= 1):
            raise ValueError("dropout rate should be between 0 and 1.")

        if not (0 <= attn_drop_rate <= 1):
            raise ValueError("attention dropout rate should be between 0 and 1.")

        if not (0 <= dropout_path_rate <= 1):
            raise ValueError("drop path rate should be between 0 and 1.")

        if feature_size % 12 != 0:
            raise ValueError("feature_size should be divisible by 12.")

        self.normalize = normalize

        self.encoder = SP_SwinTransformer(
            in_chans=in_channels,
            embed_dim=feature_size,
            window_size=window_size,
            patch_size=patch_size,
            depths=depths,
            num_heads=num_heads,
            space_embed_dim=space_embed_dim,
            mlp_ratio=4.0,
            qkv_bias=True,
            drop_rate=drop_rate,
            attn_drop_rate=attn_drop_rate,
            drop_path_rate=dropout_path_rate,
            norm_layer=nn.LayerNorm,
            use_checkpoint=use_checkpoint,
            spatial_dims=spatial_dims,
            downsample=downsample,
        )

        self.decoder = Unetr_decoder_SP(
            in_channels=in_channels,
            out_channels=out_channels,
            patch_res=patch_res,
            feature_size=feature_size,
            norm_name=norm_name,
            normalize=normalize,
            spatial_dims=spatial_dims
        )

        self.msp = nn.Parameter(torch.zeros((self.args.batch_size, self.args.size ** 2, self.args.sp_num), dtype=torch.float32), requires_grad=False)
        self.ma2 = nn.Parameter(torch.zeros((self.args.batch_size, self.args.sp_num // 4, self.args.sp_num // 16), dtype=torch.float32), requires_grad=False)
        self.ma3 = nn.Parameter(torch.zeros((self.args.batch_size, self.args.sp_num // 16, self.args.sp_num // 64), dtype=torch.float32), requires_grad=False)
        self.ma4 = nn.Parameter(torch.zeros((self.args.batch_size, self.args.sp_num // 64, self.args.sp_num // 256), dtype=torch.float32), requires_grad=False)

        self.mspt = nn.Parameter(torch.zeros((self.args.test_batch_size, self.args.size ** 2, self.args.sp_num), dtype=torch.float32), requires_grad=False)
        self.ma2t = nn.Parameter(torch.zeros((self.args.test_batch_size, self.args.sp_num // 4, self.args.sp_num // 16), dtype=torch.float32), requires_grad=False)
        self.ma3t = nn.Parameter(torch.zeros((self.args.test_batch_size, self.args.sp_num // 16, self.args.sp_num // 64), dtype=torch.float32), requires_grad=False)
        self.ma4t = nn.Parameter(torch.zeros((self.args.test_batch_size, self.args.sp_num // 64, self.args.sp_num // 256), dtype=torch.float32), requires_grad=False)

    def forward(self, x_in, matrix_sp, a1, a2, a3, a4, inference=False):

        input = x_in[:, :-1]
        mask = x_in[:, -1:]

        if inference:
            self.mspt, self.ma2t, self.ma3t, self.ma4t = _get_association(
                matrix_sp, a2, a3, a4,
                self.mspt, self.ma2t, self.ma3t, self.ma4t,
                self.args.size, self.args.sp_num, self.args.test_batch_size
            )

            hidden_states_out = self.encoder(input, self.mspt, a1, self.ma2t, self.ma3t, self.ma4t, self.normalize)
            out = self.decoder(input, hidden_states_out, self.mspt, a1, self.ma2t, self.ma3t, self.ma4t)
        else:
            self.msp, self.ma2, self.ma3, self.ma4 = _get_association(
                matrix_sp, a2, a3, a4,
                self.msp, self.ma2, self.ma3, self.ma4,
                self.args.size, self.args.sp_num, self.args.batch_size
            )

            hidden_states_out = self.encoder(input, self.msp, a1, self.ma2, self.ma3, self.ma4, self.normalize)
            out = self.decoder(input, hidden_states_out, self.msp, a1, self.ma2, self.ma3, self.ma4)

        y = torch.clamp(out, -1.0, 1.0)
        # mask
        y = y * 0.5 + 0.5
        y *= mask
        y = (y - 0.5) / 0.5

        return y

    def on_save_checkpoint(self, checkpoint):
        # pop the backbone here using custom logic
        del checkpoint['state_dict']['msp']
        del checkpoint['state_dict']['ma2']
        del checkpoint['state_dict']['ma3']
        del checkpoint['state_dict']['ma4']
        del checkpoint['state_dict']['mspt']
        del checkpoint['state_dict']['ma2t']
        del checkpoint['state_dict']['ma3t']
        del checkpoint['state_dict']['ma4t']

    def configure_optimizers(self):
        opt = torch.optim.Adam(self.parameters(), lr=self.args.lr)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=self.args.epoch)

        return [opt], [scheduler]
