
import numpy as np
import torch
from mmcv.cnn import build_conv_layer, build_norm_layer, build_upsample_layer
from mmcv.runner import BaseModule, auto_fp16
from torch import nn as nn

from mmdet.models import NECKS
import torch.nn.functional as F
import time
import pdb



@NECKS.register_module()
class SwintransformerFPN3D(BaseModule):
 
    def __init__(self,
                 in_channels=[48, 48, 96],
                 out_channels=[128, 128, 128],
                 upsample_strides=[1, 2, 4],
                 norm_cfg=dict(type='GN', num_groups=32, requires_grad=True),
                 upsample_cfg=dict(type='deconv3d', bias=False),
                 conv_cfg=dict(type='Conv3d', bias=False),
                 use_conv_for_no_stride=False,
                 use_output_upsample=False,
                 with_cp=False,
                 init_cfg=None):
        
        # replacing GN with BN3D, performance drops from 42.5 to 40.9. 
        # the difference may be exaggerated because the performance can fluncate a lot
        
        super(SwintransformerFPN3D, self).__init__(init_cfg=init_cfg)
        assert len(out_channels) == len(upsample_strides) == len(in_channels)
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.fp16_enabled = False
        self.with_cp = with_cp

        deblocks = []
        for i, out_channel in enumerate(out_channels):
            stride = upsample_strides[i]
            if stride > 1 or (stride == 1 and not use_conv_for_no_stride):
                upsample_layer = build_upsample_layer(
                    upsample_cfg,
                    in_channels=in_channels[i],
                    out_channels=out_channel,
                    kernel_size=upsample_strides[i],
                    stride=upsample_strides[i])
            else:
                stride = np.round(1 / stride).astype(np.int64)
                upsample_layer = build_conv_layer(
                    conv_cfg,
                    in_channels=in_channels[i],
                    out_channels=out_channel,
                    kernel_size=stride,
                    stride=stride)

            deblock = nn.Sequential(
                upsample_layer, build_norm_layer(norm_cfg, out_channel)[1], nn.ReLU(inplace=True))
            
            deblocks.append(deblock)
        
        self.deblocks = nn.ModuleList(deblocks)
        
        self.use_output_upsample = use_output_upsample
        if self.use_output_upsample:
            output_channel = sum(out_channels)
            self.output_deblock = nn.Sequential(
                build_upsample_layer(
                    upsample_cfg, in_channels=output_channel,
                    out_channels=output_channel, kernel_size=2, stride=2),
                build_norm_layer(norm_cfg, output_channel)[1],
                nn.ReLU(inplace=True),
                # build_conv_layer(conv_cfg, in_channels=output_channel,
                #             out_channels=output_channel, kernel_size=3, stride=1, padding=1),
                # build_norm_layer(norm_cfg, output_channel)[1],
                # nn.ReLU(inplace=True),
            )

        if init_cfg is None:
            self.init_cfg = [
                dict(type='Kaiming', layer='ConvTranspose2d'),
                dict(type='Constant', layer='NaiveSyncBatchNorm2d', val=1.0)
            ]

    @auto_fp16()
    def forward(self, x, depth):
        """Forward function.

        Args:
            x (torch.Tensor): 4D Tensor in (N, C, H, W) shape.

        Returns:
            list[torch.Tensor]: Multi-level feature maps.
        """
 

        assert len(x) == len(self.in_channels)
        ups = [deblock(x[i]) for i, deblock in enumerate(self.deblocks)]
        
        if len(ups) > 1:
            out = torch.cat(ups, dim=1) ## [4, 128, 128, 128, 16] * 3
        else:
            out = ups[0]
        
        if self.use_output_upsample: ## False
            out = torch.utils.checkpoint.checkpoint(self.output_deblock, out)

        out = F.interpolate(out, size=[ 128, 128, 16 ], mode='trilinear', align_corners=True)
        return [out]  ## [4, 384, 128, 128, 16]   B C D H W

        