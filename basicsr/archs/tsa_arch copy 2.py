'''
 Copyright 2023 xtudbxk
 Licensed under the Apache License, Version 2.0 (the "License");
 you may not use this file except in compliance with the License.
 You may obtain a copy of the License at

 http://www.apache.org/licenses/LICENSE-2.0

 Unless required by applicable law or agreed to in writing, software
 distributed under the License is distributed on an "AS IS" BASIS,
 WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 See the License for the specific language governing permissions and
 limitations under the License.
'''
import torch
from torch import nn as nn
from torch.nn import functional as F
import math
from functools import reduce
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops.layers.torch import Rearrange
from basicsr.utils.registry import ARCH_REGISTRY
from .arch_util import ResidualBlockNoBN, make_layer, ResidualBlockRCB
import math
import logging
from einops import rearrange, repeat
from einops.layers.torch import Rearrange
from functools import partial
from collections import OrderedDict
from copy import Error, deepcopy
from timm.data import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
from timm.models.layers import DropPath, to_2tuple, trunc_normal_
import torch
import torch.nn as nn
import math
from timm.models.layers import DropPath, to_2tuple, trunc_normal_
from mamba_ssm.ops.selective_scan_interface import selective_scan_fn, selective_scan_ref
from einops import rearrange, repeat
from typing import Optional, Callable
from functools import partial
from mamba_ssm import Mamba
from timm.models.layers import DropPath, to_2tuple, trunc_normal_
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import repeat
# from .tmpalign_util import TMPAlign
# from .spynet_arch import SpyNet
from spatial_correlation_sampler import SpatialCorrelationSampler
from basicsr.ops.msda import SingleScaleDeformAttnV1, SingleScaleDeformAttnV2, SingleScaleDeformAttnV3, SingleDeformAttnV4,SparseDeformAttnV4

@ARCH_REGISTRY.register()
class TSA(nn.Module):
    def __init__(self,
                 num_in_ch=3,
                 num_out_ch=3,
                 num_feat=64,
                 num_frame=15,
                 center_frame_idx=None,
                 hr_in=False):

        super().__init__()
        if center_frame_idx is None:
            self.center_frame_idx = num_frame // 2
        else:
            self.center_frame_idx = center_frame_idx
        self.hr_in = hr_in
        self.stride = 4
        self.k_num = 3
        self.keyframe_stride = 1
        self.num_feat = num_feat
        self.fastflownet = FastFlow_process('/share3/home/zqiang/TMP/experiments/pretrained_models/fastflownet_ft_mix.pth')

        # extract features for each frame
        self.conv_first = nn.Conv2d(num_in_ch, num_feat, 3, padding=1)
        self.feature_extraction = make_layer(ResidualBlockRCB, 2, num_feat=num_feat)
        self.fuse = nn.Conv2d(2*num_feat, num_feat, 3, padding=1)
        self.resblocks = make_layer(ResidualBlockRCB, 12, num_feat=num_feat)
        # align  Trajectory-Aware Alignment
        self.LTAM = LTAM(stride = self.stride, k_num = self.k_num)
        num_heads = 8
        num_points = 6
        max_residue_magnitude = 10
        self.defor_att = DeformAttnAlign(d_model=64,n_levels=1,
                                        n_heads=num_heads,n_points=num_points,
                                        max_residue_magnitude=max_residue_magnitude)

        # upsample
        # self.reconstruction = make_layer(ResidualBlockNoBN, num_reconstruct_block, num_feat=num_feat)
        self.upconv1 = nn.Conv2d(num_feat, 48, 3, 1, 1)
        self.pixel_shuffle = nn.PixelShuffle(4)
        # self.upconv1 = nn.Conv2d(num_feat, 16 * 4, 3, 1, 1, bias=True)
        # self.upconv2 = nn.Conv2d(16, 16 * 4, 3, 1, 1, bias=True)
        # self.conv_last = nn.Conv2d(16, 3, 3, 1, 1)
        # self.pixel_shuffle = nn.PixelShuffle(2)

        # activation function
        self.lrelu = nn.LeakyReLU(negative_slope=0.1, inplace=True)

    def compute_flow(self, lrs):
        """Compute optical flow using SPyNet for feature warping.
        Args:
            lrs (tensor): Input LR images with shape (n, t, c, h, w)
        Return:
            tuple(Tensor): Optical flow. 'flows_forward' corresponds to the
                flows used for forward-time propagation (current to previous).
                'flows_backward' corresponds to the flows used for
                backward-time propagation (current to next).
        """
        n, t, c, h, w = lrs.size()
        lrs_1 = lrs[:, :-1, :, :, :].reshape(-1, c, h, w)
        lrs_2 = lrs[:, 1:, :, :, :].reshape(-1, c, h, w)
        flows_forward = self.fastflownet(lrs_2, lrs_1).view(n, t - 1, 2, h, w)

        return flows_forward

    def dynamickeyframe(self, lrs, flows_forward):
        """Compute optical flow using SPyNet for feature warping.
        """
        n, t, c, h, w = lrs.size()
        # print('lrs',lrs.shape,len(flows_forward),flows_forward[0].shape)
        key_idx_list = []
        k_idx = 0
        j_idx = 0
        avg_list = [np.mean((torch.sum((flows_forward[:,j,:,:,:])**2, dim=0).sqrt()).detach().cpu().numpy()) for j in range(t-1)]
        avg_list.insert(0, 0.0)

        for i in range(0,t):
            if i == 0:
                key_idx_list.append(i)
                k_idx = k_idx + 1
            elif i == t-1 and key_idx_list[-1] != t-1:
                key_idx_list.append(i)
            else:
                start_idx = 0 if i-self.keyframe_stride-1 < 0 else i-self.keyframe_stride-1
                end_idx = t if i+self.keyframe_stride-1 > t else i+self.keyframe_stride-1
                temp_idx = avg_list.index(max(avg_list[start_idx:i]))
                # AVG_tmp = np.mean(avg_list[start_idx:i-1])
                # if avg_list[temp_idx] > AVG_tmp:
                if key_idx_list[-1] == temp_idx: # self.keyframe_stride + 1
                    continue
                else:
                    if temp_idx - key_idx_list[-1] >= self.keyframe_stride:
                        # temp_idx = avg_list.index(max(avg_list[start_idx:i]))
                        key_idx_list.append(key_idx_list[-1] + self.keyframe_stride)
                    else:
                        if avg_list[temp_idx] > avg_list[key_idx_list[-1]]:
                            key_idx_list.append(temp_idx)

        fixed_keyframe_idx = list(range(0, t, self.keyframe_stride))
        # print('key_idx_list',key_idx_list,fixed_keyframe_idx )

        return key_idx_list


    def forward(self, lrs, hidden_states=None, return_hs=False):
        n, t, c, h, w = lrs.size()
        if self.hr_in:
            assert h % 16 == 0 and w % 16 == 0, ('The height and width must be multiple of 16.')
        else:
            assert h % 4 == 0 and w % 4 == 0, ('The height and width must be multiple of 4.')

        # extract features for each frame
        feat_origin = self.lrelu(self.conv_first(lrs.view(-1, c, h, w)))
        outputs = self.feature_extraction(feat_origin).view(n, t, -1,h,w)
        outputs = torch.unbind(outputs,dim=1)
        outputs = list(outputs)

        # compute optical flow
        flows_forward = self.compute_flow(lrs)
        # keyframe_idx_forward = self.dynamickeyframe(lrs, flows_forward)
        # keyframe_idx_forward = list(range(0, t, self.keyframe_stride))

        # backward-time propgation
        # align
        fina_out = []
        memory_list = []
        feat_buffers = []
        # sparse_feat_buffers_s0 = []
        # sparse_feat_buffers_s1 = []
        memory_feat_buffers_s1 = []
        # sparse_feat_buffers_s2 = []
        # sparse_feat_buffers_s3 = []
        index_feat_buffers_s1 = []

        feat_prop = lrs.new_zeros(n, self.num_feat, h, w)
        out_feat = lrs.new_zeros(n, self.num_feat, h, w)
        grid_y, grid_x = torch.meshgrid(torch.arange(0, h//self.stride), torch.arange(0, w//self.stride))
        location_update = torch.stack([grid_x,grid_y],dim=0).type_as(lrs).expand(n,-1,-1,-1)
        for i in range(0, t):
            lr_curr = lrs[:, i, :, :, :]
            lr_curr_feat = outputs[i]

            if i > 0:  # no warping required for the first timestep
                if flows_forward is not None:
                    flow = flows_forward[:, i - 1, :, :, :]
                # feat_prop = flow_warp(feat_prop, flow.permute(0, 2, 3, 1),padding_mode='border')
                feat_prop = self.defor_att(lr_curr_feat,feat_prop,memory_list[-1],flow)

                # update the location map
                flow = F.adaptive_avg_pool2d(flow,(h//self.stride,w//self.stride))/self.stride
                location_update = flow_warp(location_update, flow.permute(0, 2, 3, 1),padding_mode='border',interpolation="nearest")# n , 2t , h , w

                feat_buffer = torch.stack(feat_buffers, dim=1)
                memory_buffer = torch.stack(memory_feat_buffers_s1, dim=1)
                # sparse_feat_buffer_s0 = torch.stack(sparse_feat_buffers_s0, dim=1)
                # sparse_feat_buffer_s1 = torch.stack(sparse_feat_buffers_s1, dim=1)
                # sparse_feat_buffer_s2 = torch.stack(sparse_feat_buffers_s2, dim=1)
                # sparse_feat_buffer_s3 = torch.stack(sparse_feat_buffers_s3, dim=1)
                index_feat_buffer_s1 = torch.stack(index_feat_buffers_s1, dim=1)
                feat_prop = self.LTAM(lr_curr_feat, index_feat_buffer_s1, feat_prop, memory_buffer, location_update)  #  sparse_feat_buffer_s2,
                # feat_prop = self.LTAM(lr_curr_feat,sparse_feat_buffer_s0, index_feat_buffer_s1,feat_prop,sparse_feat_buffer_s1,sparse_feat_buffer_s3,location_update)  #  sparse_feat_buffer_s2,
                location_update = torch.cat([location_update,torch.stack([grid_x,grid_y],dim=0).type_as(lrs).expand(n,-1,-1,-1)],dim=1)

            # feature tokenization *4
            # bs * c * h * w --> # bs * (c*4*4) * (h//4*w//4)
            # sparse_feat_prop_s1 = F.unfold(feat_prop, kernel_size=(self.stride,self.stride), padding=0, stride=self.stride)
            # # bs * (c*4*4) * (h//4*w//4) -->  bs * (c*4*4) * h//4 * w//4
            # sparse_feat_prop_s1 = F.fold(sparse_feat_prop_s1, output_size=(h//self.stride,w//self.stride), kernel_size=(1,1), padding=0, stride=1)
            # sparse_feat_buffers_s1.append(sparse_feat_prop_s1)

            # bs * c * h * w --> # bs * (c*4*4) * (h//4*w//4)
            memory_feat_prop_s1 = F.unfold(out_feat, kernel_size=(self.stride,self.stride), padding=0, stride=self.stride)
            # bs * (c*4*4) * (h//4*w//4) -->  bs * (c*4*4) * h//4 * w//4
            memory_feat_prop_s1 = F.fold(memory_feat_prop_s1, output_size=(h//self.stride,w//self.stride), kernel_size=(1,1), padding=0, stride=1)
            memory_feat_buffers_s1.append(memory_feat_prop_s1)

            # bs * c * h * w --> # bs * (c*4*4) * (h//4*w//4)
            index_feat_prop_s1 = F.unfold(lr_curr_feat, kernel_size=(self.stride,self.stride), padding=0, stride=self.stride)
            # bs * (c*4*4) * (h//4*w//4) -->  bs * (c*4*4) * h//4 * w//4
            index_feat_prop_s1 = F.fold(index_feat_prop_s1, output_size=(h//self.stride,w//self.stride), kernel_size=(1,1), padding=0, stride=1)
            index_feat_buffers_s1.append(index_feat_prop_s1)

            feat_prop = torch.cat([lr_curr_feat, feat_prop], dim=1)
            feat_prop = self.resblocks(self.fuse(feat_prop))
            feat_buffers.append(feat_prop)
            # upsampling given the backward and forward features
            out = torch.cat([lr_curr_feat,feat_prop], dim=1)
            out_feat = self.resblocks(self.fuse(out))
            memory_list.append(out_feat)

            # upsample
            out = self.pixel_shuffle(self.upconv1(out_feat)).view(n, c, 4*h, 4*w)
            base = F.interpolate(lrs[:, i, :, :, :], scale_factor=4, mode='bilinear', align_corners=False).view(n, c, 4*h, 4*w)
            out += base
            fina_out.append(out)

        del location_update
        # del sparse_feat_buffers_s0
        # del sparse_feat_buffers_s1
        del memory_feat_buffers_s1
        # del sparse_feat_buffers_s2
        # del sparse_feat_buffers_s3
        del index_feat_buffers_s1

        out_list = torch.stack(fina_out, dim=1)
        return out_list




class RoundSTE(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x):
        y = torch.bernoulli(x)
        return y

    @staticmethod
    def backward(ctx, grad):
        return grad, None





class ContextBlock(nn.Module):

    def __init__(self, n_feat, bias=False):
        super(ContextBlock, self).__init__()

        self.conv_mask = nn.Conv2d(n_feat, 1, kernel_size=1, bias=bias)
        self.softmax = nn.Softmax(dim=2)

        self.channel_add_conv = nn.Sequential(
            nn.Conv2d(n_feat, n_feat, kernel_size=1, bias=bias),
            nn.LeakyReLU(0.2),
            nn.Conv2d(n_feat, n_feat, kernel_size=1, bias=bias)
        )

    def modeling(self, x):
        batch, channel, height, width = x.size()
        input_x = x
        # [N, C, H * W]
        input_x = input_x.view(batch, channel, height * width)
        # [N, 1, C, H * W]
        input_x = input_x.unsqueeze(1)
        # [N, 1, H, W]
        context_mask = self.conv_mask(x)
        # [N, 1, H * W]
        context_mask = context_mask.view(batch, 1, height * width)
        # [N, 1, H * W]
        context_mask = self.softmax(context_mask)
        # [N, 1, H * W, 1]
        context_mask = context_mask.unsqueeze(3)
        # [N, 1, C, 1]
        context = torch.matmul(input_x, context_mask)
        # [N, C, 1, 1]
        context = context.view(batch, channel, 1, 1)

        return context

    def forward(self, x, x_r):
        # [N, C, 1, 1]
        context = self.modeling(x) + self.modeling(x_r)

        # [N, C, 1, 1]
        channel_add_term = self.channel_add_conv(context)
        x = x + channel_add_term

        return x

##########################################################################
### --------- Residual Context Block ----------
class RCB(nn.Module):
    def __init__(self, n_feat, kernel_size=3, reduction=2, bias=False, groups=1):
        super(RCB, self).__init__()

        # act = nn.LeakyReLU(0.2)

        self.body = nn.Sequential(
            nn.Conv2d(n_feat, n_feat//reduction, kernel_size=3, stride=1, padding=1, bias=bias, groups=groups),
            nn.LeakyReLU(0.2),
            nn.Conv2d(n_feat//reduction, n_feat, kernel_size=3, stride=1, padding=1, bias=bias, groups=groups)
        )

        self.act = nn.LeakyReLU(0.2)

        self.gcnet = ContextBlock(n_feat, bias=bias)
        self.gcnet1 = ContextBlock(n_feat, bias=bias)

    def forward(self, x, x_r):
        res = self.body(x)
        x_r = self.body(x_r)
        res = self.act(self.gcnet(res,x_r))
        res = self.act(self.gcnet1(res,x_r))
        res += x
        return res




class DeformAttnAlign(nn.Module):
    def __init__(self, d_model=256, n_levels=4, n_heads=8, n_points=4, max_residue_magnitude=10):
        super().__init__()
        self.ms_deform_att = SparseDeformAttnV4(d_model=d_model,n_heads=n_heads,
                                                n_points=n_points, max_residue_magnitude=max_residue_magnitude)
    def get_valid_ratio(self, mask):
        _, H, W = mask.shape
        valid_H = torch.sum(~mask[:, :, 0], 1)
        valid_W = torch.sum(~mask[:, 0, :], 1)
        valid_ratio_h = valid_H.float() / H
        valid_ratio_w = valid_W.float() / W
        valid_ratio = torch.stack([valid_ratio_w, valid_ratio_h], -1)
        return valid_ratio

    def get_reference_points(self, spatial_shapes, valid_ratios, device):
        reference_points_list = []
        for lvl, (H_, W_) in enumerate(spatial_shapes):
            ref_y, ref_x = torch.meshgrid(torch.linspace(0.5, H_ - 0.5, H_, dtype=torch.float32, device=device),
                                          torch.linspace(0.5, W_ - 0.5, W_, dtype=torch.float32, device=device))
            ref_y = ref_y.reshape(-1)[None] / (valid_ratios[:, None, lvl, 1] * H_)
            ref_x = ref_x.reshape(-1)[None] / (valid_ratios[:, None, lvl, 0] * W_)
            ref = torch.stack((ref_x, ref_y), -1)
            reference_points_list.append(ref)
        reference_points = torch.cat(reference_points_list, 1)
        reference_points = reference_points[:, :, None] * valid_ratios[:, None]
        return reference_points

    def forward(self, nbr_fea, cur_fea, ext_fea, flow):
        b, c, h, w = nbr_fea.shape
        device = nbr_fea.device

        mask = (torch.zeros(b, h, w) > 1).to(device)

        spatial_shapes = torch.as_tensor([(h, w)], dtype=torch.long).to(device)
        level_start_index = torch.cat((spatial_shapes.new_zeros((1,)), spatial_shapes.prod(1).cumsum(0)[:-1]))
        valid_ratio = torch.unsqueeze(self.get_valid_ratio(mask), dim=1)
        ref_point = self.get_reference_points(spatial_shapes, valid_ratio, device=device)

        output = self.ms_deform_att(ext_fea, ref_point, cur_fea, nbr_fea, spatial_shapes, level_start_index, input_padding_mask=mask.flatten(1), flow=flow)

        return output



class ResidualBlocksWithInputConv(nn.Module):
    """Residual blocks with a convolution in front.

    Args:
        in_channels (int): Number of input channels of the first conv.
        out_channels (int): Number of channels of the residual blocks.
            Default: 64.
        num_blocks (int): Number of residual blocks. Default: 30.
    """

    def __init__(self, in_channels, out_channels=64, num_blocks=30):
        super().__init__()

        main = []

        # a convolution used to match the channels of the residual blocks
        main.append(nn.Conv2d(in_channels, out_channels, 3, 1, 1, bias=True))
        main.append(nn.LeakyReLU(negative_slope=0.1, inplace=True))

        # residual blocks
        main.append(
            make_layer(
                ResidualBlockNoBN, num_blocks, num_feat=out_channels))

        self.main = nn.Sequential(*main)

    def forward(self, feat):
        """
        Forward function for ResidualBlocksWithInputConv.

        Args:
            feat (Tensor): Input feature with shape (n, in_channels, h, w)

        Returns:
            Tensor: Output feature with shape (n, out_channels, h, w)
        """
        return self.main(feat)



class Scale_Selection(nn.Module):
    def __init__(self, dim):
        super(Scale_Selection, self).__init__()
        self.getscale = nn.Sequential(
            nn.Conv2d(2 + dim, dim, 1, 1, 0, bias=True),
            nn.Conv2d(dim, dim, 1, 2, 0, bias=True),
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(dim, 2),
            nn.Sigmoid()
        )


    def forward(self, x, location_feat):
        scale = RoundSTE.apply(self.getscale( torch.cat((x,location_feat),1))).unsqueeze(2).unsqueeze(3)
        return scale



class Scale_Adaptive_Agg(nn.Module):
    def __init__(self, dim):
        super(Scale_Adaptive_Agg, self).__init__()
        self.initConv =nn.Conv2d(dim, dim, 3, 1, 1, bias=True)
        self.initLConv =nn.Conv2d(2, 2, 3, 1, 1, bias=True)
        self.Fusion = make_layer(ResidualBlockNoBN, 5, num_feat=dim)
        self.Scale_Agg = Scale_Agg(dim=dim)

    def forward(self, x, flow, location_feat, scale):
        feat = x

        scale_tmp = 1
        x0 = self.initConv(x)
        location_feat = self.initLConv(location_feat)
        x0 = self.Scale_Agg(x0, flow, location_feat, scale_tmp)
        feat = self.Fusion(feat) #  + x0 * scale[:, 0:1] * scale[:, 1:2]

        # if scale.shape[0] != 1 or (scale[:, 0:1].mean() >= 0.5 and scale[:, 1:2].mean() >= 0.5):
        #     scale_tmp = 1
        #     x0 = self.initConv(x)
        #     location_feat = self.initLConv(location_feat)
        #     x0 = self.Scale_Agg(x0, flow, location_feat, scale_tmp)
        #     feat = self.Fusion(feat + x0 * scale[:, 0:1] * scale[:, 1:2])

        # if scale.shape[0] != 1 or (scale[:, 0:1].mean() < 0.5 and scale[:, 1:2].mean() >= 0.5):
        #     x1 = self.initConv(F.interpolate(x, scale_factor=0.5, mode="bilinear"))
        #     location_feat1 =self.initLConv(F.interpolate(location_feat, scale_factor=0.5, mode="bilinear"))
        #     scale_tmp = 2
        #     x1 = self.Scale_Agg(x1, flow, location_feat1, scale_tmp)
        #     feat = self.Fusion(feat + F.interpolate(x1, scale_factor=2.0, mode="bilinear") * (1 - scale[:, 0:1]) * scale[:, 1:2])

        # if scale.shape[0] != 1 or scale[:, 1:2].mean() < 0.5:
        #     x2 = self.initConv(F.interpolate(x, scale_factor=0.25, mode="bilinear"))
        #     location_feat2 = self.initLConv(F.interpolate(location_feat, scale_factor=0.25, mode="bilinear"))
        #     scale_tmp = 4
        #     x2 = self.Scale_Agg(x2, flow, location_feat2, scale_tmp)
        #     feat = self.Fusion(feat + F.interpolate(x2, scale_factor=4.0, mode="bilinear") * (1 - scale[:, 1:2]))

        return feat



class Rectract_Adaptive(nn.Module):
    def __init__(self, dim):
        super(Rectract_Adaptive, self).__init__()
        self.initConv =nn.Conv2d(dim, dim, 3, 1, 1, bias=True)
        self.initLConv =nn.Conv2d(2, 2, 3, 1, 1, bias=True)
        self.Fusion = make_layer(ResidualBlockNoBN, 5, num_feat=dim)
        self.RecDeform_Agg = RectractDeform_Agg(dim=dim)

    def forward(self, feat_prop, x, flow, location_feat, scale):
        feat = x
        scale_tmp = 1
        x0 = self.initConv(x)
        #  scale 4 3 2 1 | 4 3 2 1 |
        location_feat = self.initLConv(location_feat)
        x0 = self.RecDeform_Agg(feat_prop, x0, flow, location_feat, scale_tmp)
        feat = self.Fusion(feat) #  + x0 * scale[:, 0:1] * scale[:, 1:2]

        return feat





class RectractDeform_Agg(nn.Module):
    def __init__(self, dim):
        super(RectractDeform_Agg, self).__init__()
        self.stride = 0
        num_heads = 4
        num_points = 2
        max_residue_magnitude = 10
        num_levels = 1

    def forward(self, feat_prop, lr_curr_feat, flow, location_update, scale):
        self.stride = scale
        # print('scale',self.stride, lr_curr_feat.shape)

        B, C, H, W = lr_curr_feat.shape
        if H%self.stride!=0:
            pad_h = self.stride - H%self.stride
            lr_curr_feat = F.pad(lr_curr_feat, (0, 0, 0, pad_h), 'reflect')
            flow = F.pad(flow, (0, 0, 0, pad_h), 'reflect')
            location_update = F.pad(location_update, (0, 0, 0, pad_h), 'reflect')
        if W%self.stride!=0:
            pad_w = self.stride - W%self.stride
            lr_curr_feat = F.pad(lr_curr_feat, (0, pad_w, 0, 0), 'reflect')
            flow = F.pad(flow, (0, pad_w, 0, 0), 'reflect')
            location_update = F.pad(location_update, (0, pad_w, 0, 0), 'reflect')

        # print('lr_curr_feat',lr_curr_feat.shape)
        B, C, h, w = lr_curr_feat.shape
        # curr_out = lr_curr_feat.view(B, h, w, C)
        extra_feat = torch.cat([lr_curr_feat, flow_warp(feat_prop, flow.permute(0, 2, 3, 1))], dim=1)

        # feat_prop = self.flow_guided_dcn(feat_prop, lr_curr_feat, extra_feat, flow)
        # lr_curr_feat = align_out # .view(B, C, h, w)

        index_feat_set_s1 = F.unfold(lr_curr_feat, kernel_size=(self.stride,self.stride), padding=0, stride=self.stride)
        index_feat_set_s1 = F.fold(index_feat_set_s1, output_size=(h//self.stride,w//self.stride), kernel_size=(1,1), padding=0, stride=1)

        feat_len = int(C*self.stride*self.stride)
        feat_num = int((h//self.stride) * (w//self.stride))

        # grid_flow [0,h-1][0,w-1] -> [-1,1][-1,1]
        flow = F.adaptive_avg_pool2d(flow,(h//self.stride,w//self.stride))/self.stride
        location_update = F.interpolate(location_update, scale_factor=1/self.stride, mode="bilinear")
        location_update = flow_warp(location_update, flow.permute(0, 2, 3, 1),padding_mode='border',interpolation="nearest")# n , 2t , h , w

        grid_flow = location_update.contiguous().view(B,2,h//self.stride,w//self.stride).permute(0, 2, 3, 1)
        grid_flow_x = 2.0 * grid_flow[:, :, :, 0] / max(w//self.stride - 1, 1) - 1.0
        grid_flow_y = 2.0 * grid_flow[:, :, :, 1] / max(h//self.stride - 1, 1) - 1.0
        grid_flow = torch.stack((grid_flow_x.unsqueeze(1), grid_flow_y.unsqueeze(1)), dim=4)

        output_s1 = F.grid_sample(lr_curr_feat.contiguous().view(-1,(C*self.stride*self.stride),(h//self.stride),(w//self.stride)),grid_flow.contiguous().view(-1,(h//self.stride),(w//self.stride),2),mode='nearest',padding_mode='zeros',align_corners=True) # (nt) * (c*4*4) * (h//4) * (w//4)

        index_output_s1 = F.grid_sample(index_feat_set_s1.contiguous().view(-1,(C*self.stride*self.stride),(h//self.stride),(w//self.stride)),grid_flow.contiguous().view(-1,(h//self.stride),(w//self.stride),2),mode='nearest',padding_mode='zeros',align_corners=True) # (nt) * (c*4*4) * (h//4) * (w//4)
        # n * c * h * w --> # n * (c*4*4) * (h//4*w//4)
        curr_feat = F.unfold(lr_curr_feat, kernel_size=(self.stride,self.stride), padding=0, stride=self.stride)
        curr_feat = curr_feat.permute(0, 2, 1)
        curr_feat = F.normalize(curr_feat, dim=2).unsqueeze(3) # n * (h//4*w//4) * (c*4*4) * 1

        ## update index map for soft corr
        index_output_s1 = index_output_s1.contiguous().view(B*1,(C*self.stride*self.stride),(h//self.stride),(w//self.stride))
        index_output_s1 = F.unfold(index_output_s1, kernel_size=(1, 1), padding=0, stride=1).view(B,-1,feat_len,feat_num)
        index_output_s1 = index_output_s1.permute(0, 3, 1, 2)
        index_output_s1 = F.normalize(index_output_s1, dim=3) # n * (h//4*w//4) * t * (c*4*4)

        matrix_index = torch.matmul(index_output_s1, curr_feat).squeeze(3) # n * (h//4*w//4) * t
        matrix_index = matrix_index.view(B,feat_num,1)# n * (h//4*w//4) * t
        #### Top-1
        corr_soft, corr_index = torch.max(matrix_index, dim=2)# n * (h//4*w//4)
        corr_soft00, corr_index00 = torch.topk(matrix_index, k=1, dim=2)# n * (h//4*w//4)
        print(corr_soft==corr_soft00, corr_index==corr_index00)
        corr_soft = corr_soft.unsqueeze(1).expand(-1,feat_len,-1)
        corr_soft = F.fold(corr_soft, output_size=(h,w), kernel_size=(self.stride,self.stride), padding=0, stride=self.stride)

        #### Top-2
        corr_soft2, corr_index2 = torch.topk(matrix_index, k=2, dim=2)# n * (h//4*w//4)
        corr_soft2 = corr_soft2.unsqueeze(1).expand(-1,feat_len,-1)
        corr_soft2 = F.fold(corr_soft2, output_size=(h,w), kernel_size=(self.stride,self.stride), padding=0, stride=self.stride)

         #### Top-3
        corr_soft3, corr_index3 = torch.topk(matrix_index, k=3, dim=2)# n * (h//4*w//4)
        corr_soft3 = corr_soft3.unsqueeze(1).expand(-1,feat_len,-1)
        corr_soft3 = F.fold(corr_soft3, output_size=(h,w), kernel_size=(self.stride,self.stride), padding=0, stride=self.stride)



        # Aggr
        # n * t * (c*4*4) * h//4 * w//4 --> nt * (c*4*4) * h//4 * w//4
        output_s1 = output_s1.contiguous().view(B*1,(C*self.stride*self.stride),(h//self.stride),(w//self.stride))
        output_s1 = F.unfold(output_s1, kernel_size=(1, 1), padding=0, stride=1).view(B,-1,feat_len,feat_num)
        output_s1 = torch.gather(output_s1.contiguous().view(B,1,feat_len,feat_num), 1, corr_index.view(B,1,1,feat_num).expand(-1,-1,feat_len,-1))
        output_s1 = output_s1.squeeze(1)
        output_s1 = F.fold(output_s1, output_size=(h,w), kernel_size=(self.stride,self.stride), padding=0, stride=self.stride)

        out = output_s1 * corr_soft + lr_curr_feat
        extra_feat = torch.cat([out, flow_warp(feat_prop, flow.permute(0, 2, 3, 1))], dim=1)
        # feat_prop = self.flow_guided_dcn(feat_prop, lr_curr_feat, extra_feat, flow)
        out = self.flow_guided_dcn(feat_prop, out, extra_feat, flow)
        out = out[:,:,:H, :W]

        return out




class Scale_Agg(nn.Module):
    def __init__(self, dim):
        super(Scale_Agg, self).__init__()
        self.stride = 0
        self.win_size = 2
        num_heads = 2
        mlp_ratio= 4.0
        drop=0
        attn_drop=0.
        qkv_bias=True
        qk_scale=None
        token_projection = 'linear'
        self.shift_size = 0
        # self.Sparse_Attention  = Sparse_Attention(dim=dim)
        self.spa_attn = WindowAttention_sparse(
                    dim, win_size=to_2tuple(self.win_size), num_heads=num_heads,
                    qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop,
                    token_projection=token_projection)


    def forward(self, lr_curr_feat, flow, location_update, scale):
        self.stride = scale
        # print('scale',self.stride, lr_curr_feat.shape)

        B, C, H, W = lr_curr_feat.shape
        if H%self.stride!=0:
            pad_h = self.stride - H%self.stride
            lr_curr_feat = F.pad(lr_curr_feat, (0, 0, 0, pad_h), 'reflect')
            flow = F.pad(flow, (0, 0, 0, pad_h), 'reflect')
            location_update = F.pad(location_update, (0, 0, 0, pad_h), 'reflect')
        if W%self.stride!=0:
            pad_w = self.stride - W%self.stride
            lr_curr_feat = F.pad(lr_curr_feat, (0, pad_w, 0, 0), 'reflect')
            flow = F.pad(flow, (0, pad_w, 0, 0), 'reflect')
            location_update = F.pad(location_update, (0, pad_w, 0, 0), 'reflect')

        # print('lr_curr_feat',lr_curr_feat.shape)
        B, C, h, w = lr_curr_feat.shape
        curr_out = lr_curr_feat.view(B, h, w, C)

        # cyclic shift
        if self.shift_size > 0:
            shifted_curr = torch.roll(curr_out, shifts=(-self.shift_size, -self.shift_size), dims=(1, 2))
        else:
            shifted_curr = curr_out

        # partition windows
        out_windows = window_partition(shifted_curr, self.win_size)  # nW*B, win_size, win_size, C  N*C->C
        out_windows = out_windows.view(-1, self.win_size * self.win_size, C)  # nW*B, win_size*win_size, C
        attn_windows = self.spa_attn(out_windows)
        # merge windows
        attn_windows = attn_windows.view(-1, self.win_size, self.win_size, C)
        shifted_x = window_reverse(attn_windows, self.win_size, h, w)  # B H' W' C

        # reverse cyclic shift
        if self.shift_size > 0:
            curr_out = torch.roll(shifted_x, shifts=(self.shift_size, self.shift_size), dims=(1, 2))
        else:
            curr_out = shifted_x
        lr_curr_feat = curr_out.view(B, C, h, w)

        index_feat_set_s1 = F.unfold(lr_curr_feat, kernel_size=(self.stride,self.stride), padding=0, stride=self.stride)
        index_feat_set_s1 = F.fold(index_feat_set_s1, output_size=(h//self.stride,w//self.stride), kernel_size=(1,1), padding=0, stride=1)

        feat_len = int(C*self.stride*self.stride)
        feat_num = int((h//self.stride) * (w//self.stride))

        # grid_flow [0,h-1][0,w-1] -> [-1,1][-1,1]
        flow = F.adaptive_avg_pool2d(flow,(h//self.stride,w//self.stride))/self.stride
        location_update = F.interpolate(location_update, scale_factor=1/self.stride, mode="bilinear")
        location_update = flow_warp(location_update, flow.permute(0, 2, 3, 1),padding_mode='border',interpolation="nearest")# n , 2t , h , w

        grid_flow = location_update.contiguous().view(B,2,h//self.stride,w//self.stride).permute(0, 2, 3, 1)
        grid_flow_x = 2.0 * grid_flow[:, :, :, 0] / max(w//self.stride - 1, 1) - 1.0
        grid_flow_y = 2.0 * grid_flow[:, :, :, 1] / max(h//self.stride - 1, 1) - 1.0
        grid_flow = torch.stack((grid_flow_x.unsqueeze(1), grid_flow_y.unsqueeze(1)), dim=4)

        output_s1 = F.grid_sample(lr_curr_feat.contiguous().view(-1,(C*self.stride*self.stride),(h//self.stride),(w//self.stride)),grid_flow.contiguous().view(-1,(h//self.stride),(w//self.stride),2),mode='nearest',padding_mode='zeros',align_corners=True) # (nt) * (c*4*4) * (h//4) * (w//4)

        index_output_s1 = F.grid_sample(index_feat_set_s1.contiguous().view(-1,(C*self.stride*self.stride),(h//self.stride),(w//self.stride)),grid_flow.contiguous().view(-1,(h//self.stride),(w//self.stride),2),mode='nearest',padding_mode='zeros',align_corners=True) # (nt) * (c*4*4) * (h//4) * (w//4)
        # n * c * h * w --> # n * (c*4*4) * (h//4*w//4)
        curr_feat = F.unfold(lr_curr_feat, kernel_size=(self.stride,self.stride), padding=0, stride=self.stride)
        curr_feat = curr_feat.permute(0, 2, 1)
        curr_feat = F.normalize(curr_feat, dim=2).unsqueeze(3) # n * (h//4*w//4) * (c*4*4) * 1

        ## update index map for soft corr
        index_output_s1 = index_output_s1.contiguous().view(B*1,(C*self.stride*self.stride),(h//self.stride),(w//self.stride))
        index_output_s1 = F.unfold(index_output_s1, kernel_size=(1, 1), padding=0, stride=1).view(B,-1,feat_len,feat_num)
        index_output_s1 = index_output_s1.permute(0, 3, 1, 2)
        index_output_s1 = F.normalize(index_output_s1, dim=3) # n * (h//4*w//4) * t * (c*4*4)
        ## Sparse information
        # index_output_s1 = self.SABlock(index_output_s1.unsqueeze(1)).squeeze(1)
        # curr_feat = self.SABlock(curr_feat.unsqueeze(1)).squeeze(1)
        # print('index_output_s1',index_output_s1.shape, curr_feat.shape)  # [4, 4096, 1, 64]  [4, 4096, 64, 1]
        # curr_feat = self.Sparse_Attention(curr_feat.squeeze(3))["x"].unsqueeze(3)
        matrix_index = torch.matmul(index_output_s1, curr_feat).squeeze(3) # n * (h//4*w//4) * t
        # print('matrix_index',matrix_index.shape) # [4, 4096, 1]
        matrix_index = matrix_index.view(B,feat_num,1)# n * (h//4*w//4) * t
        corr_soft, corr_index = torch.max(matrix_index, dim=2)# n * (h//4*w//4)
        corr_soft = corr_soft.unsqueeze(1).expand(-1,feat_len,-1)
        corr_soft = F.fold(corr_soft, output_size=(h,w), kernel_size=(self.stride,self.stride), padding=0, stride=self.stride)

        # Aggr
        # n * t * (c*4*4) * h//4 * w//4 --> nt * (c*4*4) * h//4 * w//4
        output_s1 = output_s1.contiguous().view(B*1,(C*self.stride*self.stride),(h//self.stride),(w//self.stride))
        output_s1 = F.unfold(output_s1, kernel_size=(1, 1), padding=0, stride=1).view(B,-1,feat_len,feat_num)
        output_s1 = torch.gather(output_s1.contiguous().view(B,1,feat_len,feat_num), 1, corr_index.view(B,1,1,feat_num).expand(-1,-1,feat_len,-1))
        output_s1 = output_s1.squeeze(1)
        output_s1 = F.fold(output_s1, output_size=(h,w), kernel_size=(self.stride,self.stride), padding=0, stride=self.stride)

        out = output_s1 * corr_soft + lr_curr_feat
        # print('out',out.shape)
        # out = self.Sparse_Attention(out.view(B,C,-1))["x"]
        # out = out.view(B,C,h,w)[:,:,:H, :W]

        out = out.view(B, h, w, C)

        # cyclic shift
        if self.shift_size > 0:
            shifted_out = torch.roll(out, shifts=(-self.shift_size, -self.shift_size), dims=(1, 2))
        else:
            shifted_out = out

        # partition windows
        out_windows = window_partition(shifted_out, self.win_size)  # nW*B, win_size, win_size, C  N*C->C
        out_windows = out_windows.view(-1, self.win_size * self.win_size, C)  # nW*B, win_size*win_size, C
        attn_windows = self.spa_attn(out_windows)
        # merge windows
        attn_windows = attn_windows.view(-1, self.win_size, self.win_size, C)
        shifted_x = window_reverse(attn_windows, self.win_size, h, w)  # B H' W' C

        # reverse cyclic shift
        if self.shift_size > 0:
            out = torch.roll(shifted_x, shifts=(self.shift_size, self.shift_size), dims=(1, 2))
        else:
            out = shifted_x
        out = out.view(B, C, h, w)
        out = out[:,:,:H, :W]

        return out




class LTAM(nn.Module):
    def __init__(self, stride=4, k_num=3):
        super().__init__()
        self.stride = stride
        self.k_num = k_num
        self.fusion = nn.Conv2d(3 * 64, 64, 1, 1, 0, bias=True)
        # self.RCB = RCB(64, reduction=1)
        num_heads = 8
        num_points = 6
        max_residue_magnitude = 10
        split_list = np.linspace(0.5, 1, num = self.k_num+1) # np.arange(0,0.5,self.k_num+1)
        # self.defor_att = nn.ModuleList([DeformAttnAlign(d_model=64, n_levels=1,
        #                                 n_heads=num_heads, n_points=int(num_points*split_list[i+1]),
        #                                 max_residue_magnitude=max_residue_magnitude) for i in range(self.k_num)])

        self.defor_att = DeformAttnAlign(d_model=64, n_levels=1,
                                        n_heads=num_heads, n_points=num_points,
                                        max_residue_magnitude=max_residue_magnitude)
    def forward(self, curr_feat,  index_feat_set_s1, anchor_feat, memory_buffer_set, location_feat):
        """Compute the long-range trajectory-aware attention.  sparse_feat_set_s2,
        Args:
            anchor_feat (tensor): Input feature with shape (n, c, h, w)
            sparse_feat_set_s0 (tensor): Input tokens with shape (n, t, c*4*4, h//4, w//4)
            sparse_feat_set_s1 (tensor): Input tokens with shape (n, t, c*4*4, h//4, w//4)
            sparse_feat_set_s2 (tensor): Input tokens with shape (n, t, c*4*4, h//4, w//4)
            sparse_feat_set_s3 (tensor): Input tokens with shape (n, t, c*4*4, h//4, w//4)
            location_feat (tensor): Input location map with shape (n, 2*t, h//4, w//4)
        Return:
            fusion_feature (tensor): Output fusion feature with shape (n, c, h, w).
        """

        n, c, h, w = anchor_feat.size()
        t = memory_buffer_set.size(1)
        feat_len = int(c*self.stride*self.stride)
        feat_num = int((h//self.stride) * (w//self.stride))
        feat_curr0 = curr_feat

        # grid_flow [0,h-1][0,w-1] -> [-1,1][-1,1]
        grid_flow = location_feat.contiguous().view(n,t,2,h//self.stride,w//self.stride).permute(0, 1, 3, 4, 2)
        grid_flow_x = 2.0 * grid_flow[:, :, :, :, 0] / max(w//self.stride - 1, 1) - 1.0
        grid_flow_y = 2.0 * grid_flow[:, :, :, :, 1] / max(h//self.stride - 1, 1) - 1.0
        grid_flow = torch.stack((grid_flow_x, grid_flow_y), dim=4)

        # output_s1 = F.grid_sample(sparse_feat_set_s1.contiguous().view(-1,(c*self.stride*self.stride),(h//self.stride),(w//self.stride)),grid_flow.contiguous().view(-1,(h//self.stride),(w//self.stride),2),mode='nearest',padding_mode='zeros',align_corners=True) # (nt) * (c*4*4) * (h//4) * (w//4)
        # # n * t * (c*4*4) * h//4 * w//4 --> nt * (c*4*4) * h//4 * w//4
        # output_s1_v = (output_s1).contiguous().view(n*t,(c*self.stride*self.stride),(h//self.stride),(w//self.stride))
        # # nt * (c*4*4) * h//4 * w//4 --> n * t * (c*4*4) * (h//4*w//4)
        # output_s1_u = F.unfold(output_s1_v, kernel_size=(1, 1), padding=0, stride=1).view(n,-1,feat_len,feat_num)

        memory_s1 = F.grid_sample(memory_buffer_set.contiguous().view(-1,(c*self.stride*self.stride),(h//self.stride),(w//self.stride)),grid_flow.contiguous().view(-1,(h//self.stride),(w//self.stride),2),mode='nearest',padding_mode='zeros',align_corners=True) # (nt) * (c*4*4) * (h//4) * (w//4)
        # n * t * (c*4*4) * h//4 * w//4 --> nt * (c*4*4) * h//4 * w//4
        memory_s1_v = (memory_s1).contiguous().view(n*t,(c*self.stride*self.stride),(h//self.stride),(w//self.stride))
        # nt * (c*4*4) * h//4 * w//4 --> n * t * (c*4*4) * (h//4*w//4)
        memory_s1_u = F.unfold(memory_s1_v, kernel_size=(1, 1), padding=0, stride=1).view(n,-1,feat_len,feat_num)

        index_output_s1 = F.grid_sample(index_feat_set_s1.contiguous().view(-1,(c*self.stride*self.stride),(h//self.stride),(w//self.stride)),grid_flow.contiguous().view(-1,(h//self.stride),(w//self.stride),2),mode='nearest',padding_mode='zeros',align_corners=True) # (nt) * (c*4*4) * (h//4) * (w//4)
        # n * c * h * w --> # n * (c*4*4) * (h//4*w//4)
        curr_feat = F.unfold(curr_feat, kernel_size=(self.stride,self.stride), padding=0, stride=self.stride)
        # n * (c*4*4) * (h//4*w//4) --> n * (h//4*w//4) * (c*4*4)
        curr_feat = curr_feat.permute(0, 2, 1)
        curr_feat = F.normalize(curr_feat, dim=2).unsqueeze(3) # n * (h//4*w//4) * (c*4*4) * 1

        # cross-scale attention * 4
        # n * t * (c*4*4) * h//4 * w//4 --> nt * (c*4*4) * h//4 * w//4
        index_output_s1 = index_output_s1.contiguous().view(n*t,(c*self.stride*self.stride),(h//self.stride),(w//self.stride))
        # nt * (c*4*4) * h//4 * w//4 --> n * t * (c*4*4) * (h//4*w//4)
        index_output_s1 = F.unfold(index_output_s1, kernel_size=(1, 1), padding=0, stride=1).view(n,-1,feat_len,feat_num)
        # n * t * (c*4*4) * (h//4*w//4) --> n * (h//4*w//4) * t * (c*4*4)
        index_output_s1 = index_output_s1.permute(0, 3, 1, 2)
        index_output_s1 = F.normalize(index_output_s1, dim=3) # n * (h//4*w//4) * t * (c*4*4)
        # [ n * (h//4*w//4) * t * (c*4*4) ]  *  [ n * (h//4*w//4) * (c*4*4) * 1 ]  -->  n * (h//4*w//4) * t
        matrix_index = torch.matmul(index_output_s1, curr_feat).squeeze(3) # n * (h//4*w//4) * t
        matrix_index = matrix_index.view(n,feat_num,t)# n * (h//4*w//4) * t
        # corr_soft, corr_index = torch.max(matrix_index, dim=2)# n * (h//4*w//4)
        # n * (h//4*w//4) --> n * (c*4*4) * (h//4*w//4)

        corr_soft_list = []
        corr_index_list = []
        if self.k_num <= t:
            corr_soft_k, corr_index_k = torch.topk(matrix_index, k=self.k_num, dim=2)  # n * (h//4*w//4)
            [corr_soft_list.append(corr_soft_k[:,:,i]) for i in range(self.k_num)][0]
            [corr_index_list.append(corr_index_k[:,:,i]) for i in range(self.k_num)][0]
        else:
            corr_soft, corr_index = torch.max(matrix_index, dim=2)  # n * (h//4*w//4)
            [corr_soft_list.append(corr_soft) for i in range(self.k_num)][0]
            [corr_index_list.append(corr_index)  for i in range(self.k_num)][0]

        out = anchor_feat.new_zeros(n, c, h, w)
        out_list = []
        for ii in range(self.k_num-1,-1,-1):
            corr_soft = corr_soft_list[ii]
            corr_soft = corr_soft.unsqueeze(1).expand(-1,feat_len,-1)
            # n * (c*4*4) * (h//4*w//4) --> n * c * h * w
            corr_soft = F.fold(corr_soft, output_size=(h,w), kernel_size=(self.stride,self.stride), padding=0, stride=self.stride)
            corr_index = corr_index_list[ii]

            # # n * t * (c*4*4) * (h//4*w//4) --> n * 1 * (c*4*4) * (h//4*w//4)
            # output_s1_g = torch.gather(output_s1_u.contiguous().view(n,t,feat_len,feat_num), 1, corr_index.view(n,1,1,feat_num).expand(-1,-1,feat_len,-1))# n * 1 * (c*4*4) * (h//4*w//4)
            # # n * 1 * (c*4*4) * (h//4*w//4)  --> n * (c*4*4) * (h//4*w//4)
            # output_s1_s = output_s1_g.squeeze(1)
            # # # n * (c*4*4) * (h//4*w//4) --> n * c * h * w
            # output_s1_f = F.fold(output_s1_s, output_size=(h,w), kernel_size=(self.stride,self.stride), padding=0, stride=self.stride)
            # n * t * (c*4*4) * (h//4*w//4) --> n * 1 * (c*4*4) * (h//4*w//4)
            memory_s1_g = torch.gather(memory_s1_u.contiguous().view(n,t,feat_len,feat_num), 1, corr_index.view(n,1,1,feat_num).expand(-1,-1,feat_len,-1))# n * 1 * (c*4*4) * (h//4*w//4)
            # n * 1 * (c*4*4) * (h//4*w//4)  --> n * (c*4*4) * (h//4*w//4)
            memory_s1_s = memory_s1_g.squeeze(1)
            # # n * (c*4*4) * (h//4*w//4) --> n * c * h * w
            memory_s1_f = F.fold(memory_s1_s, output_size=(h,w), kernel_size=(self.stride,self.stride), padding=0, stride=self.stride)

            # out_list.append(output_s1_f+memory_s1_f)
            # output_s1_f = [out + out_list[i] for i in range(len(out_list))][0]
            output_agg = self.fusion(torch.cat([memory_s1_f + out, feat_curr0, anchor_feat], dim=1))
            output_agg = self.defor_att(output_agg, feat_curr0, corr_soft, None)
            # output_agg = output_agg * corr_soft
            out = out + output_agg * corr_soft

        return out





########### window-based self-attention #############
class WindowAttention_sparse(nn.Module):
    def __init__(self, dim, win_size,num_heads, token_projection='linear', qkv_bias=True, qk_scale=None, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.dim = dim
        self.win_size = win_size  # Wh, Ww
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5

        # define a parameter table of relative position bias
        self.relative_position_bias_table = nn.Parameter(
            torch.zeros((2 * win_size[0] - 1) * (2 * win_size[1] - 1), num_heads))  # 2*Wh-1 * 2*Ww-1, nH

        # get pair-wise relative position index for each token inside the window
        coords_h = torch.arange(self.win_size[0]) # [0,...,Wh-1]
        coords_w = torch.arange(self.win_size[1]) # [0,...,Ww-1]
        coords = torch.stack(torch.meshgrid([coords_h, coords_w]))  # 2, Wh, Ww
        coords_flatten = torch.flatten(coords, 1)  # 2, Wh*Ww
        relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]  # 2, Wh*Ww, Wh*Ww
        relative_coords = relative_coords.permute(1, 2, 0).contiguous()  # Wh*Ww, Wh*Ww, 2
        relative_coords[:, :, 0] += self.win_size[0] - 1  # shift to start from 0
        relative_coords[:, :, 1] += self.win_size[1] - 1
        relative_coords[:, :, 0] *= 2 * self.win_size[1] - 1
        relative_position_index = relative_coords.sum(-1)  # Wh*Ww, Wh*Ww
        self.register_buffer("relative_position_index", relative_position_index)
        trunc_normal_(self.relative_position_bias_table, std=.02)

        if token_projection =='linear':
            self.qkv = LinearProjection(dim,num_heads,dim//num_heads,bias=qkv_bias)
        else:
            raise Exception("Projection error!")

        self.token_projection = token_projection
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

        self.softmax = nn.Softmax(dim=-1)
        self.relu = nn.ReLU()
        self.w = nn.Parameter(torch.ones(2))

    def forward(self, x, attn_kv=None, mask=None):
        B_, N, C = x.shape
        q, k, v = self.qkv(x,attn_kv)
        q = q * self.scale
        attn = (q @ k.transpose(-2, -1))

        relative_position_bias = self.relative_position_bias_table[self.relative_position_index.view(-1)].view(
            self.win_size[0] * self.win_size[1], self.win_size[0] * self.win_size[1], -1)  # Wh*Ww,Wh*Ww,nH
        relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous()  # nH, Wh*Ww, Wh*Ww
        ratio = attn.size(-1)//relative_position_bias.size(-1)
        relative_position_bias = repeat(relative_position_bias, 'nH l c -> nH l (c d)', d = ratio)

        attn = attn + relative_position_bias.unsqueeze(0)

        if mask is not None:
            nW = mask.shape[0]
            mask = repeat(mask, 'nW m n -> nW m (n d)',d = ratio)
            attn = attn.view(B_ // nW, nW, self.num_heads, N, N*ratio) + mask.unsqueeze(1).unsqueeze(0)
            attn = attn.view(-1, self.num_heads, N, N*ratio)
            attn0 = self.softmax(attn)
            attn1 = self.relu(attn)**2#b,h,w,c
        else:
            attn0 = self.softmax(attn)
            attn1 = self.relu(attn)**2
        w1 = torch.exp(self.w[0]) / torch.sum(torch.exp(self.w))
        w2 = torch.exp(self.w[1]) / torch.sum(torch.exp(self.w))
        attn = attn0*w1+attn1*w2
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B_, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x

    def extra_repr(self) -> str:
        return f'dim={self.dim}, win_size={self.win_size}, num_heads={self.num_heads}'



class LinearProjection(nn.Module):
    def __init__(self, dim, heads = 8, dim_head = 64, dropout = 0., bias=True):
        super().__init__()
        inner_dim = dim_head *  heads
        self.heads = heads
        self.to_q = nn.Linear(dim, inner_dim, bias = bias)
        self.to_kv = nn.Linear(dim, inner_dim * 2, bias = bias)
        self.dim = dim
        self.inner_dim = inner_dim

    def forward(self, x, attn_kv=None):
        B_, N, C = x.shape
        if attn_kv is not None:
            attn_kv = attn_kv.unsqueeze(0).repeat(B_,1,1)
        else:
            attn_kv = x
        N_kv = attn_kv.size(1)
        q = self.to_q(x).reshape(B_, N, 1, self.heads, C // self.heads).permute(2, 0, 3, 1, 4)
        kv = self.to_kv(attn_kv).reshape(B_, N_kv, 2, self.heads, C // self.heads).permute(2, 0, 3, 1, 4)
        q = q[0]
        k, v = kv[0], kv[1]
        return q,k,v



########### window operation#############
def window_partition(x, win_size, dilation_rate=1):
    B, H, W, C = x.shape
    if dilation_rate !=1:
        x = x.permute(0,3,1,2) # B, C, H, W
        assert type(dilation_rate) is int, 'dilation_rate should be a int'
        x = F.unfold(x, kernel_size=win_size,dilation=dilation_rate,padding=4*(dilation_rate-1),stride=win_size) # B, C*Wh*Ww, H/Wh*W/Ww
        windows = x.permute(0,2,1).contiguous().view(-1, C, win_size, win_size) # B' ,C ,Wh ,Ww
        windows = windows.permute(0,2,3,1).contiguous() # B' ,Wh ,Ww ,C
    else:
        x = x.view(B, H // win_size, win_size, W // win_size, win_size, C)
        windows = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, win_size, win_size, C) # B' ,Wh ,Ww ,C
    return windows

def window_reverse(windows, win_size, H, W, dilation_rate=1):
    # B' ,Wh ,Ww ,C
    B = int(windows.shape[0] / (H * W / win_size / win_size))
    x = windows.view(B, H // win_size, W // win_size, win_size, win_size, -1)
    if dilation_rate !=1:
        x = windows.permute(0,5,3,4,1,2).contiguous() # B, C*Wh*Ww, H/Wh*W/Ww
        x = F.fold(x, (H, W), kernel_size=win_size, dilation=dilation_rate, padding=4*(dilation_rate-1),stride=win_size)
    else:
        x = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(B, H, W, -1)
    return x

#########################################



def compute_sparsity(x):
    total_num = torch.numel(x)
    num_non_zero = torch.count_nonzero(x)
    num_zero = total_num - num_non_zero
    sparsity = num_zero / total_num
    return sparsity




class MaskPredictor(nn.Module):
    """ Mask Predictor using Low rank MHA"""
    def __init__(self,
                 dim,
                 num_heads=8,
                 num_tokens=24,
                 attn_keep_rate=0.25,
                 reduce_n_factor=8,
                 reduce_c_factor=2,
                 share_inout_proj=False,
                 qk_scale=None
                 ):
        super().__init__()
        self.num_tokens = num_tokens
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.reduced_c = self.head_dim // reduce_c_factor
        self.reduced_n = int(num_tokens // reduce_n_factor)
        self.scale = qk_scale or self.num_heads ** -0.5
        self.proj_c_q = nn.Linear(self.head_dim, self.reduced_c)
        self.proj_c_k = nn.Linear(self.head_dim, self.reduced_c)

        self.proj_n = nn.Parameter(torch.zeros(self.num_tokens, self.reduced_n))
        # trunc_normal_(self.proj_back_n, std=.02, a=0.)
        trunc_normal_(self.proj_n, std=.02)
        if share_inout_proj:
            self.proj_back_n = self.proj_n
        else:
            self.proj_back_n = nn.Parameter(torch.zeros(self.num_tokens, self.reduced_n))
            trunc_normal_(self.proj_back_n, std=.02)

        self.basis_threshold = nn.Threshold(2e-2, 0.)
        self.basis_coef_threshold = nn.Threshold(5e-2, 0.)

        self.attn_budget = math.ceil(attn_keep_rate * num_tokens)

    def forward(self, q, k, token_mask=None):
        # TODO: Perform full self-attention if attn_budget > token_budget
        B, H, N, C = q.shape
        # self.num_tokens = N

        if token_mask is not None:
            token_budget = token_mask[0].sum(dim=-1)
            self.attn_budget = token_budget if token_budget < self.attn_budget else self.attn_budget

        out_dict = {}

        B, H, N, C = q.shape
        print('N',N, self.num_tokens)
        assert self.num_tokens == N
        q, k = self.proj_c_q(q), self.proj_c_k(k)  # [B, H, N, c]
        if token_mask is not None:
            # token_mask: [B, N-1]
            q[..., 1:, :] = q[..., 1:, :].masked_fill(~token_mask[:, None, :, None], 0.)
            k[..., 1:, :] = k[..., 1:, :].masked_fill(~token_mask[:, None, :, None], 0.)

        k = k.permute(0, 1, 3, 2)  # [B, H, c, N]
        k = k @ self.proj_n  # [B, H, c, k]

        # TODO: should call this only once during inference.
        # if self.training and self.cfg.LOSS.USE_ATTN_RECON:
        #     basis = self.proj_back_n.permute(1, 0)
        # else:
        basis = self.proj_back_n.permute(1, 0)
        # basis[basis.abs() <= cfg.BASIS_THRESHOLD] = 0.
        # For Linear attention visualization
        basis = self.basis_threshold(basis.abs())

        # Compute low-rank approximation of the attention matrix
        # q: [B, H, N, C]   k: [B, H, c, K]
        cheap_attn = (q @ k) * self.scale  # [B, H, N, K]
        cheap_attn = cheap_attn[..., 1:, :]  # [B, H, N-1, K] remove cls token
        basis_coef = cheap_attn.softmax(dim=-1)  # [B, H, N-1, K] coef is naturally sparse
        # if self.training and self.cfg.LOSS.USE_ATTN_RECON:
            # approx_attn = basis_coef @ basis  # [B, H, N-1, N]
        # if cfg.BASIS_COEF.USE_TOPK:
        basis_coef_topk, basis_coef_topk_indices = basis_coef.topk(8, sorted=False)
        basis_coef = torch.zeros_like(basis_coef, device=basis_coef.device)
        basis_coef.scatter_(-1, basis_coef_topk_indices, basis_coef_topk)
        # elif cfg.BASIS_COEF.THRESHOLD > 0:
        #     # basis_coef[basis_coef <= cfg.BASIS_COEF.THRESHOLD] = 0.
        #     basis_coef = self.basis_coef_threshold(basis_coef)
        approx_attn = basis_coef @ basis  # [B, H, N-1, N]

        # Zero out attention connectivity columns corresponding to inactive tokens
        attn_score = approx_attn.clone()  # [B, H, N-1, N]
        if token_mask is not None:
            attn_score[..., 1:].masked_fill_(~token_mask[:, None, None, :], float('-inf'))  # [B, H, N-1, N]

        # Generate columns of instance dependent sparse attention connectivity pattern
        # if cfg.ATTN_SCORE.USE_TOPK:
        # Top-k attention connectivity
        topk_cont_indices = torch.topk(attn_score, self.attn_budget, sorted=False)[1]  # [B, H, N-1, num_cont]
        attn_mask = torch.zeros_like(attn_score, dtype=attn_score.dtype, device=attn_score.device)
        attn_mask.scatter_(-1, topk_cont_indices, True)  # [B, H, N-1, N]
        # elif cfg.ATTN_SCORE.THRESHOLD > 0:
        #     # Threshold attention connectivity
        #     attn_mask = torch.where(attn_score <= cfg.ATTN_SCORE.THRESHOLD, 0., 1.)
        # else:
        #     raise NotImplementedError

        # Zero out attention connectivity rows corresponding to inactive tokens
        if token_mask is not None:
            attn_mask *= token_mask[:, None, :, None]  # [B, H, N-1, N]

        # Add cls token back to attn mask
        cls_mask = torch.ones(B, H, 1, N, dtype=attn_mask.dtype, device=attn_mask.device)
        attn_mask = torch.cat([cls_mask, attn_mask], dim=2)  # [B, H, N, N]
        attn_mask.detach_()  # TODO: No gradient for attn_mask

        out_dict['basis_coef'] = basis_coef
        out_dict['approx_attn'] = approx_attn
        out_dict['attn_mask'] = attn_mask
        if not self.training:
            if cfg.OUT_BASIS_SPARSITY:
                out_dict['basis_sparsity'] = compute_sparsity(basis)
            if cfg.OUT_BASIS_COEF_SPARSITY:
                out_dict['basis_coef_sparsity'] = compute_sparsity(basis_coef)
            if cfg.OUT_ATTN_MASK_SPARSITY:
                out_dict['attn_mask_sparsity'] = compute_sparsity(attn_mask)
        return out_dict


class Sparse_Attention(nn.Module):
    def __init__(
            self,
            dim,
            num_tokens=64*64,
            num_heads=4,
            attn_keep_rate=0.25,
            token_keep_rate=0.50,
            token_pruning_this_layer=False,
            reduce_n_factor=8,
            reduce_c_factor=2,
            share_inout_proj=False,
            qkv_bias=False,
            qk_scale=None,
            attn_drop=0.0,
            proj_drop=0.0,
    ):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5
        # print('dim',dim)

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

        self.mask_predictor = MaskPredictor(
            dim,
            num_heads=num_heads,
            num_tokens=num_tokens,
            attn_keep_rate=attn_keep_rate,
            reduce_n_factor=reduce_n_factor,
            reduce_c_factor=reduce_c_factor,
            share_inout_proj=share_inout_proj,
        )
        self.token_pruning_this_layer = token_pruning_this_layer
        self.token_keep_rate = token_keep_rate
        self.token_budget = math.ceil(token_keep_rate * (num_tokens - 1))

    def softmax_with_policy(self, attn, policy, eps=1e-6):
        # https://discuss.pytorch.org/t/how-to-implement-the-exactly-same-softmax-as-f-softmax-by-pytorch/44263/9
        B, H, N, N = attn.size()
        attn_policy = policy
        max_att = torch.max(attn, dim=-1, keepdim=True)[0]
        attn = attn - max_att
        # attn = attn.exp_() * attn_policy
        # return attn / attn.sum(dim=-1, keepdim=True)

        # for stable training
        attn = attn.to(torch.float32).exp_() * attn_policy.to(torch.float32)
        attn = (attn + eps / N) / (attn.sum(dim=-1, keepdim=True) + eps)
        return attn.type_as(max_att)

    def forward(self, x, prev_token_mask=None):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        # q, k, v shape of [B, H, N, C]

        # Zero out key query values corresponding to inactive tokens
        if prev_token_mask is not None and not self.cfg.LOSS.USE_ATTN_RECON:
            q[..., 1:, :] = q[..., 1:, :].masked_fill(~prev_token_mask[:, None, :, None], 0.)
            k[..., 1:, :] = k[..., 1:, :].masked_fill(~prev_token_mask[:, None, :, None], 0.)
            v[..., 1:, :] = v[..., 1:, :].masked_fill(~prev_token_mask[:, None, :, None], 0.)

        out_dict = self.mask_predictor(q, k, prev_token_mask)
        attn_mask = out_dict['attn_mask']

        attn = (q @ k.transpose(-2, -1)) * self.scale
        unmasked_attn = attn.clone().softmax(dim=-1)
        # if self.training and self.cfg.LOSS.USE_ATTN_RECON:
        #     attn = attn.softmax(dim=-1)  # Don't distort the token value when reconstructing attention
        # else:
        # attn = self.softmax_with_policy(attn, attn_mask)
        attn.masked_fill_(~attn_mask.bool(), float('-inf'))
        attn = attn.softmax(dim=-1)
        attn = torch.nan_to_num(attn)  # Some rows are pruned and filled with '-inf'

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)

        out_dict['token_mask'] = prev_token_mask
        # if self.token_pruning_this_layer and not self.cfg.LOSS.USE_ATTN_RECON:  # TODO: refactor this
        cls_attn = attn[:, :, 0, 1:]  # [B, H, N-1]
        token_score = cls_attn.mean(dim=1)  # [B, N-1]
        if prev_token_mask is not None:
            token_score = token_score.masked_fill(~prev_token_mask, float('-inf'))
        topk_token_indices = torch.topk(token_score, self.token_budget, sorted=False)[1]  # [B, left_tokens]
        new_token_mask = torch.zeros_like(token_score, dtype=torch.bool, device=token_score.device)
        new_token_mask.scatter_(-1, topk_token_indices, True)  # [B, N-1]
        out_dict['token_mask'] = new_token_mask  # TODO: would masked_fill be faster than indices fill?

        new_val = {'x': x, 'masked_attn': attn, 'unmasked_attn': unmasked_attn}
        out_dict.update(new_val)
        return out_dict





def flow_warp(x,
              flow,
              interpolation='bilinear',
              padding_mode='zeros',
              align_corners=True):
    """Warp an image or a feature map with optical flow.

    Args:
        x (Tensor): Tensor with size (n, c, h, w).
        flow (Tensor): Tensor with size (n, h, w, 2). The last dimension is
            a two-channel, denoting the width and height relative offsets.
            Note that the values are not normalized to [-1, 1].
        interpolation (str): Interpolation mode: 'nearest' or 'bilinear'.
            Default: 'bilinear'.
        padding_mode (str): Padding mode: 'zeros' or 'border' or 'reflection'.
            Default: 'zeros'.
        align_corners (bool): Whether align corners. Default: True.

    Returns:
        Tensor: Warped image or feature map.
    """
    if x.size()[-2:] != flow.size()[1:3]:
        raise ValueError(f'The spatial sizes of input ({x.size()[-2:]}) and '
                         f'flow ({flow.size()[1:3]}) are not the same.')
    _, _, h, w = x.size()
    # create mesh grid
    grid_y, grid_x = torch.meshgrid(torch.arange(0, h), torch.arange(0, w))
    grid = torch.stack((grid_x, grid_y), 2).type_as(x)  # (w, h, 2)
    grid.requires_grad = False

    grid_flow = grid + flow
    # scale grid_flow to [-1,1]
    grid_flow_x = 2.0 * grid_flow[:, :, :, 0] / max(w - 1, 1) - 1.0
    grid_flow_y = 2.0 * grid_flow[:, :, :, 1] / max(h - 1, 1) - 1.0
    grid_flow = torch.stack((grid_flow_x, grid_flow_y), dim=3)
    output = F.grid_sample(
        x,
        grid_flow,
        mode=interpolation,
        padding_mode=padding_mode,
        align_corners=align_corners)
    return output






class Correlation(nn.Module):
    def __init__(self, max_displacement):
        super(Correlation, self).__init__()
        self.max_displacement = max_displacement
        self.kernel_size = 2*max_displacement+1
        self.corr = SpatialCorrelationSampler(1, self.kernel_size, 1, 0, 1)

    def forward(self, x, y):
        b, c, h, w = x.shape
        return self.corr(x, y).view(b, -1, h, w) / c




def convrelu(in_channels, out_channels, kernel_size=3, stride=1, padding=1, dilation=1, groups=1, bias=True):
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, dilation, groups, bias=bias),
        nn.LeakyReLU(0.1, inplace=True)
    )




def deconv(in_planes, out_planes, kernel_size=4, stride=2, padding=1):
    return nn.ConvTranspose2d(in_planes, out_planes, kernel_size, stride, padding, bias=True)




class Decoder(nn.Module):
    def __init__(self, in_channels, groups):
        super(Decoder, self).__init__()
        self.in_channels = in_channels
        self.groups = groups
        self.conv1 = convrelu(in_channels, 96, 3, 1)
        self.conv2 = convrelu(96, 96, 3, 1, groups=groups)
        self.conv3 = convrelu(96, 96, 3, 1, groups=groups)
        self.conv4 = convrelu(96, 96, 3, 1, groups=groups)
        self.conv5 = convrelu(96, 64, 3, 1)
        self.conv6 = convrelu(64, 32, 3, 1)
        self.conv7 = nn.Conv2d(32, 2, 3, 1, 1)


    def channel_shuffle(self, x, groups):
        b, c, h, w = x.size()
        channels_per_group = c // groups
        x = x.view(b, groups, channels_per_group, h, w)
        x = x.transpose(1, 2).contiguous()
        x = x.view(b, -1, h, w)
        return x


    def forward(self, x):
        if self.groups == 1:
            out = self.conv7(self.conv6(self.conv5(self.conv4(self.conv3(self.conv2(self.conv1(x)))))))
        else:
            out = self.conv1(x)
            out = self.channel_shuffle(self.conv2(out), self.groups)
            out = self.channel_shuffle(self.conv3(out), self.groups)
            out = self.channel_shuffle(self.conv4(out), self.groups)
            out = self.conv7(self.conv6(self.conv5(out)))
        return out



class MambaLayerlocal(nn.Module):
    def __init__(self, dim, d_state=16, d_conv=4, expand=2, mlp_ratio=4, drop=0., drop_path=0., act_layer=nn.GELU,
                 reverse=True):
        super().__init__()
        self.dim = dim
        self.norm1 = nn.LayerNorm(dim)
        self.mamba = Mamba(
            d_model=dim,  # Model dimension d_model
            d_state=d_state,  # SSM state expansion factor
            d_conv=d_conv,  # Local convolution width
            expand=expand,  # Block expansion factor
            bimamba_type="v2",
            # use_fast_path=False,
        )

        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = nn.LayerNorm(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)
        self.reverse = reverse
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            fan_out //= m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()

    def forward(self, x, hilbert_curve):

        x = x.permute(0, 2, 1, 3, 4)
        B, C, nf, H, W = x.shape

        if self.reverse:
            x = x.permute(0, 1, 3, 4, 2)
        assert C == self.dim

        img_dims = x.shape[2:]
        x_hw = x.flatten(2).contiguous()

        x_hil = x_hw.index_select(dim=-1, index=hilbert_curve)
        x_flat = x_hil.transpose(-1, -2)

        # Bi-Mamba layer
        x_mamba = x_flat + self.drop_path(self.mamba(self.norm1(x_flat)))
        x_mamba_out = x_mamba + self.drop_path(self.mlp(self.norm2(x_mamba), nf, H, W))
        outmamba = x_mamba_out.transpose(-1, -2)

        sum_out = torch.zeros_like(outmamba)
        hilbert_curve_re = repeat(hilbert_curve, 'hw -> b c hw', b=outmamba.shape[0], c=outmamba.shape[1])
        assert outmamba.shape == hilbert_curve_re.shape

        sum_out.scatter_add_(dim=-1, index=hilbert_curve_re, src=outmamba)
        sum_out = sum_out.reshape(B, C, *img_dims).contiguous()

        if self.reverse:
            out = sum_out.permute(0, 1, 4, 2, 3)

        out = out.permute(0, 2, 1, 3, 4)

        return out



class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class SS2D(nn.Module):
    def __init__(
            self,
            d_model,
            d_state=16,
            d_conv=3,
            expand=2.,
            dt_rank="auto",
            dt_min=0.001,
            dt_max=0.1,
            dt_init="random",
            dt_scale=1.0,
            dt_init_floor=1e-4,
            dropout=0.,
            conv_bias=True,
            bias=False,
            device=None,
            dtype=None,
            **kwargs,
    ):
        factory_kwargs = {"device": device, "dtype": dtype}
        super().__init__()
        self.d_model = d_model
        self.d_state = d_state
        self.d_conv = d_conv
        self.expand = expand
        self.d_inner = int(self.expand * self.d_model)
        self.dt_rank = math.ceil(self.d_model / 16) if dt_rank == "auto" else dt_rank

        self.in_proj = nn.Linear(self.d_model, self.d_inner * 2, bias=bias, **factory_kwargs)
        self.conv2d = nn.Conv2d(
            in_channels=self.d_inner,
            out_channels=self.d_inner,
            groups=self.d_inner,
            bias=conv_bias,
            kernel_size=d_conv,
            padding=(d_conv - 1) // 2,
            **factory_kwargs,
        )
        self.act = nn.SiLU()

        self.x_proj = (
            nn.Linear(self.d_inner, (self.dt_rank + self.d_state * 2), bias=False, **factory_kwargs),
            nn.Linear(self.d_inner, (self.dt_rank + self.d_state * 2), bias=False, **factory_kwargs),
            nn.Linear(self.d_inner, (self.dt_rank + self.d_state * 2), bias=False, **factory_kwargs),
            nn.Linear(self.d_inner, (self.dt_rank + self.d_state * 2), bias=False, **factory_kwargs),
        )
        self.x_proj_weight = nn.Parameter(torch.stack([t.weight for t in self.x_proj], dim=0))
        del self.x_proj

        self.dt_projs = (
            self.dt_init(self.dt_rank, self.d_inner, dt_scale, dt_init, dt_min, dt_max, dt_init_floor,
                         **factory_kwargs),
            self.dt_init(self.dt_rank, self.d_inner, dt_scale, dt_init, dt_min, dt_max, dt_init_floor,
                         **factory_kwargs),
            self.dt_init(self.dt_rank, self.d_inner, dt_scale, dt_init, dt_min, dt_max, dt_init_floor,
                         **factory_kwargs),
            self.dt_init(self.dt_rank, self.d_inner, dt_scale, dt_init, dt_min, dt_max, dt_init_floor,
                         **factory_kwargs),
        )
        self.dt_projs_weight = nn.Parameter(torch.stack([t.weight for t in self.dt_projs], dim=0))
        self.dt_projs_bias = nn.Parameter(torch.stack([t.bias for t in self.dt_projs], dim=0))
        del self.dt_projs

        self.A_logs = self.A_log_init(self.d_state, self.d_inner, copies=4, merge=True)
        self.Ds = self.D_init(self.d_inner, copies=4, merge=True)

        self.selective_scan = selective_scan_fn

        self.out_norm = nn.LayerNorm(self.d_inner)
        self.out_proj = nn.Linear(self.d_inner, self.d_model, bias=bias, **factory_kwargs)
        self.dropout = nn.Dropout(dropout) if dropout > 0. else None

    @staticmethod
    def dt_init(dt_rank, d_inner, dt_scale=1.0, dt_init="random", dt_min=0.001, dt_max=0.1, dt_init_floor=1e-4,
                **factory_kwargs):
        dt_proj = nn.Linear(dt_rank, d_inner, bias=True, **factory_kwargs)

        dt_init_std = dt_rank ** -0.5 * dt_scale
        if dt_init == "constant":
            nn.init.constant_(dt_proj.weight, dt_init_std)
        elif dt_init == "random":
            nn.init.uniform_(dt_proj.weight, -dt_init_std, dt_init_std)
        else:
            raise NotImplementedError


        dt = torch.exp(
            torch.rand(d_inner, **factory_kwargs) * (math.log(dt_max) - math.log(dt_min))
            + math.log(dt_min)
        ).clamp(min=dt_init_floor)
        inv_dt = dt + torch.log(-torch.expm1(-dt))
        with torch.no_grad():
            dt_proj.bias.copy_(inv_dt)

        dt_proj.bias._no_reinit = True

        return dt_proj

    @staticmethod
    def A_log_init(d_state, d_inner, copies=1, device=None, merge=True):
        A = repeat(
            torch.arange(1, d_state + 1, dtype=torch.float32, device=device),
            "n -> d n",
            d=d_inner,
        ).contiguous()
        A_log = torch.log(A)
        if copies > 1:
            A_log = repeat(A_log, "d n -> r d n", r=copies)
            if merge:
                A_log = A_log.flatten(0, 1)
        A_log = nn.Parameter(A_log)
        A_log._no_weight_decay = True
        return A_log

    @staticmethod
    def D_init(d_inner, copies=1, device=None, merge=True):
        D = torch.ones(d_inner, device=device)
        if copies > 1:
            D = repeat(D, "n1 -> r n1", r=copies)
            if merge:
                D = D.flatten(0, 1)
        D = nn.Parameter(D)
        D._no_weight_decay = True
        return D

    def merge_x(self, x):
        B, C, H, W = x.shape
        L = 2 * H * W
        x = x.view(B, -1, L//2).transpose(1, 2)
        x = torch.cat([x[:B//2], x[B//2:]], dim=-1).reshape(B//2, L, C)
        return x.transpose(1, 2).contiguous()

    def forward_core(self, x: torch.Tensor):
        B, C, H, W = x.shape
        L = 2 * H * W
        K = 4
        B = B // 2
        x_hwwh = torch.stack([self.merge_x(x), self.merge_x(torch.transpose(x, dim0=2, dim1=3).contiguous())], dim=1).view(B, 2, -1, L)
        xs = torch.cat([x_hwwh, torch.flip(x_hwwh, dims=[-1])], dim=1)
        x_dbl = torch.einsum("b k d l, k c d -> b k c l", xs.view(B, K, -1, L), self.x_proj_weight)
        dts, Bs, Cs = torch.split(x_dbl, [self.dt_rank, self.d_state, self.d_state], dim=2)
        dts = torch.einsum("b k r l, k d r -> b k d l", dts.view(B, K, -1, L), self.dt_projs_weight)
        xs = xs.float().view(B, -1, L)
        dts = dts.contiguous().float().view(B, -1, L)
        Bs = Bs.float().view(B, K, -1, L)
        Cs = Cs.float().view(B, K, -1, L)
        Ds = self.Ds.float().view(-1)
        As = -torch.exp(self.A_logs.float()).view(-1, self.d_state)
        dt_projs_bias = self.dt_projs_bias.float().view(-1)
        out_y = self.selective_scan(
            xs, dts,
            As, Bs, Cs, Ds, z=None,
            delta_bias=dt_projs_bias,
            delta_softplus=True,
            return_last_state=False,
        ).view(B, K, -1, L)
        assert out_y.dtype == torch.float

        inv_y = torch.flip(out_y[:, 2:4], dims=[-1]).view(B, 2, -1, L)

        wh_y = torch.transpose(out_y[:, 1].view(B, -1, W, H), dim0=2, dim1=3).contiguous().view(B, -1, L)
        invwh_y = torch.transpose(inv_y[:, 1].view(B, -1, W, H), dim0=2, dim1=3).contiguous().view(B, -1, L)

        return out_y[:, 0], inv_y[:, 0], wh_y, invwh_y

    def forward(self, x: torch.Tensor, **kwargs):
        B, H, W, C = x.shape
        xz = self.in_proj(x)
        x, z = xz.chunk(2, dim=-1)
        x = x.permute(0, 3, 1, 2).contiguous()
        x = self.act(self.conv2d(x))
        # print(x.shape)
        y1, y2, y3, y4 = self.forward_core(x)
        assert y1.dtype == torch.float32
        y = y1 + y2 + y3 + y4
        # print(y.shape)
        y = torch.transpose(y, dim0=1, dim1=2).contiguous().view(B//2, H*W, 2, int(self.expand*C))
        y = torch.cat([y[:, :, 0], y[:, :, 1]], 0).view(B, H, W, int(self.expand*C))#.view(B//2, 2*H, 2*W, -1)


        y = self.out_norm(y)
        y = y * F.silu(z)
        out = self.out_proj(y)

        if self.dropout is not None:
            out = self.dropout(out)
        return out


class VSSBlock(nn.Module):
    def __init__(
            self,
            hidden_dim: int = 0,
            drop_path: float = 0,
            norm_layer: Callable[..., torch.nn.Module] = nn.LayerNorm,
            attn_drop_rate: float = 0,
            d_state: int = 16,
            mlp_ratio: float = 2.,
            **kwargs,
    ):
        super().__init__()
        self.ln_1 = norm_layer(hidden_dim)
        self.self_attention = SS2D(d_model=hidden_dim, d_state=d_state,expand=mlp_ratio,dropout=attn_drop_rate, **kwargs)
        self.skip_scale= nn.Parameter(torch.ones(hidden_dim))
        self.conv_blk = CAB(hidden_dim)
        self.ln_2 = nn.LayerNorm(hidden_dim)
        self.skip_scale2 = nn.Parameter(torch.ones(hidden_dim))


    def forward(self, input):
        input = input.permute(0, 2, 3, 1).contiguous()
        x = self.ln_1(input)

        x = input * self.skip_scale + self.self_attention(x)
        x = x * self.skip_scale2 + self.conv_blk(self.ln_2(x).permute(0, 3, 1, 2).contiguous()).permute(0, 2, 3, 1).contiguous()

        return x.permute(0, 3, 1, 2).contiguous()


class BiMambaBlock(nn.Module):
    def __init__(self,
                 dim,
                 depth,
                 norm_layer=nn.LayerNorm,
                 downsample=None,
                 use_checkpoint=False):

        super().__init__()
        self.dim = dim
        self.depth = depth
        self.blocks = nn.ModuleList()
        for _ in range(depth):
            self.blocks.append(VSSBlock(
                hidden_dim=dim,
                norm_layer=nn.LayerNorm,
                d_state=16,
            ))


    def forward(self, x):
        for blk in self.blocks:
            x = blk(x)
        return x


class OverlapPatchEmbed(nn.Module):
    def __init__(self, patch_size=7, stride=4, in_chans=3, embed_dim=768):
        super().__init__()
        patch_size = to_2tuple(patch_size)

        self.patch_size = patch_size
        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=stride,
                              padding=(patch_size[0] // 2, patch_size[1] // 2))
        self.norm = nn.LayerNorm(in_chans)

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            fan_out //= m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()

    def forward(self, x):
        B, C, H, W = x.shape
        x = x.permute(0, 2, 3, 1).contiguous()
        x = self.norm(x).permute(0, 3, 1, 2).contiguous()
        x = self.proj(x)
        return x


class ConvBlock(nn.Module):
    def __init__(self, in_dim, out_dim, depths=2,act_layer=nn.PReLU):
        super().__init__()
        layers = []
        for i in range(depths):
            if i == 0:
                layers.append(nn.Conv2d(in_dim, out_dim, 3,1,1))
            else:
                layers.append(nn.Conv2d(out_dim, out_dim, 3,1,1))
            layers.extend([
                act_layer(out_dim),
            ])
        self.conv = nn.Sequential(*layers)

    def _init_weights(self, m):
        if isinstance(m, nn.Conv2d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            fan_out //= m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()

    def forward(self, x):
        x = self.conv(x)
        return x




class FastFlowNet(nn.Module):
    def __init__(self, groups=3):
        super(FastFlowNet, self).__init__()
        self.groups = groups
        self.pconv1_1 = convrelu(3, 16, 3, 2)
        self.pconv1_2 = convrelu(16, 16, 3, 1)
        self.pconv2_1 = convrelu(16, 32, 3, 2)
        self.pconv2_2 = convrelu(32, 32, 3, 1)
        self.pconv2_3 = convrelu(32, 32, 3, 1)
        self.pconv3_1 = convrelu(32, 64, 3, 2)
        self.pconv3_2 = convrelu(64, 64, 3, 1)
        self.pconv3_3 = convrelu(64, 64, 3, 1)

        self.corr = Correlation(4)
        self.index = torch.tensor([0, 2, 4, 6, 8,
                10, 12, 14, 16,
                18, 20, 21, 22, 23, 24, 26,
                28, 29, 30, 31, 32, 33, 34,
                36, 38, 39, 40, 41, 42, 44,
                46, 47, 48, 49, 50, 51, 52,
                54, 56, 57, 58, 59, 60, 62,
                64, 66, 68, 70,
                72, 74, 76, 78, 80])

        self.rconv2 = convrelu(32, 32, 3, 1)
        self.rconv3 = convrelu(64, 32, 3, 1)
        self.rconv4 = convrelu(64, 32, 3, 1)
        self.rconv5 = convrelu(64, 32, 3, 1)
        self.rconv6 = convrelu(64, 32, 3, 1)

        self.up3 = deconv(2, 2)
        self.up4 = deconv(2, 2)
        self.up5 = deconv(2, 2)
        self.up6 = deconv(2, 2)

        self.decoder2 = Decoder(87, groups)
        self.decoder3 = Decoder(87, groups)
        self.decoder4 = Decoder(87, groups)
        self.decoder5 = Decoder(87, groups)
        self.decoder6 = Decoder(87, groups)

        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
                nn.init.kaiming_normal_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)


    def warp(self, x, flo):
        B, C, H, W = x.size()
        xx = torch.arange(0, W).view(1, -1).repeat(H, 1)
        yy = torch.arange(0, H).view(-1, 1).repeat(1, W)
        xx = xx.view(1, 1, H, W).repeat(B, 1, 1, 1)
        yy = yy.view(1, 1, H, W).repeat(B, 1, 1, 1)
        grid = torch.cat([xx, yy], 1).to(x)
        vgrid = grid + flo
        vgrid[:, 0, :, :] = 2.0 * vgrid[:, 0, :, :] / max(W-1, 1) - 1.0
        vgrid[:, 1, :, :] = 2.0 * vgrid[:, 1, :, :] / max(H-1, 1) - 1.0
        vgrid = vgrid.permute(0, 2, 3, 1)
        output = F.grid_sample(x, vgrid, mode='bilinear', align_corners=True)
        return output


    def forward(self, x):
        img1 = x[:, :3, :, :]
        img2 = x[:, 3:6, :, :]
        f11 = self.pconv1_2(self.pconv1_1(img1))
        f21 = self.pconv1_2(self.pconv1_1(img2))
        f12 = self.pconv2_3(self.pconv2_2(self.pconv2_1(f11)))
        f22 = self.pconv2_3(self.pconv2_2(self.pconv2_1(f21)))
        f13 = self.pconv3_3(self.pconv3_2(self.pconv3_1(f12)))
        f23 = self.pconv3_3(self.pconv3_2(self.pconv3_1(f22)))
        f14 = F.avg_pool2d(f13, kernel_size=(2, 2), stride=(2, 2))
        f24 = F.avg_pool2d(f23, kernel_size=(2, 2), stride=(2, 2))
        f15 = F.avg_pool2d(f14, kernel_size=(2, 2), stride=(2, 2))
        f25 = F.avg_pool2d(f24, kernel_size=(2, 2), stride=(2, 2))
        f16 = F.avg_pool2d(f15, kernel_size=(2, 2), stride=(2, 2))
        f26 = F.avg_pool2d(f25, kernel_size=(2, 2), stride=(2, 2))

        flow7_up = torch.zeros(f16.size(0), 2, f16.size(2), f16.size(3)).to(f15)
        cv6 = torch.index_select(self.corr(f16, f26), dim=1, index=self.index.to(f16).long())
        r16 = self.rconv6(f16)
        cat6 = torch.cat([cv6, r16, flow7_up], 1)
        flow6 = self.decoder6(cat6)

        flow6_up = self.up6(flow6)
        f25_w = self.warp(f25, flow6_up*0.625)
        cv5 = torch.index_select(self.corr(f15, f25_w), dim=1, index=self.index.to(f15).long())
        r15 = self.rconv5(f15)
        cat5 = torch.cat([cv5, r15, flow6_up], 1)
        flow5 = self.decoder5(cat5) + flow6_up

        flow5_up = self.up5(flow5)
        f24_w = self.warp(f24, flow5_up*1.25)
        cv4 = torch.index_select(self.corr(f14, f24_w), dim=1, index=self.index.to(f14).long())
        r14 = self.rconv4(f14)
        cat4 = torch.cat([cv4, r14, flow5_up], 1)
        flow4 = self.decoder4(cat4) + flow5_up

        flow4_up = self.up4(flow4)
        f23_w = self.warp(f23, flow4_up*2.5)
        cv3 = torch.index_select(self.corr(f13, f23_w), dim=1, index=self.index.to(f13).long())
        r13 = self.rconv3(f13)
        cat3 = torch.cat([cv3, r13, flow4_up], 1)
        flow3 = self.decoder3(cat3) + flow4_up

        flow3_up = self.up3(flow3)
        f22_w = self.warp(f22, flow3_up*5.0)
        cv2 = torch.index_select(self.corr(f12, f22_w), dim=1, index=self.index.to(f12).long())
        r12 = self.rconv2(f12)
        cat2 = torch.cat([cv2, r12, flow3_up], 1)
        flow2 = self.decoder2(cat2) + flow3_up

        # if self.training:
        #     return flow2, flow3, flow4, flow5, flow6
        # else:
        return flow2




class FastFlow_process(nn.Module):
    """FastFlow_process architecture.
    Args:
        load_path (str): path for pretrained FastFlow_process. Default: None.
    """

    def __init__(self, load_path=None):
        super(FastFlow_process, self).__init__()
        # self.basic_module = nn.ModuleList([BasicModule() for _ in range(6)])
        self.model = FastFlowNet().cuda().eval()
        if load_path:
            self.model.load_state_dict(torch.load(load_path))

        self.div_flow = 20.0
        self.div_size = 64
        # self.register_buffer('mean', torch.Tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
        # self.register_buffer('std', torch.Tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))


    def centralize(self, img1, img2):
        b, c, h, w = img1.shape
        rgb_mean = torch.cat([img1, img2], dim=2).view(b, c, -1).mean(2).view(b, c, 1, 1)
        return img1 - rgb_mean, img2 - rgb_mean, rgb_mean

    def forward(self, img1, img2):
        assert img1.size() == img2.size()

        # img1 = torch.from_numpy(cv2.imread(img1_path)).float().permute(2, 0, 1).unsqueeze(0)/255.0
        # img2 = torch.from_numpy(cv2.imread(img2_path)).float().permute(2, 0, 1).unsqueeze(0)/255.0
        img1, img2, _ = self.centralize(img1, img2)

        height, width = img1.shape[-2:]
        orig_size = (int(height), int(width))

        if height % self.div_size != 0 or width % self.div_size != 0:
            input_size = (
                int(self.div_size * np.ceil(height / self.div_size)),
                int(self.div_size * np.ceil(width / self.div_size))
            )
            img1 = F.interpolate(img1, size=input_size, mode='bilinear', align_corners=False)
            img2 = F.interpolate(img2, size=input_size, mode='bilinear', align_corners=False)
        else:
            input_size = orig_size

        input_t = torch.cat([img1, img2], 1).cuda()

        output = self.model(input_t) # .data
        # print('output',output.shape)

        flow = self.div_flow * F.interpolate(output, size=input_size, mode='bilinear', align_corners=False)

        if input_size != orig_size:
            scale_h = orig_size[0] / input_size[0]
            scale_w = orig_size[1] / input_size[1]
            flow = F.interpolate(flow, size=orig_size, mode='bilinear', align_corners=False)
            flow[:, 0, :, :] *= scale_w
            flow[:, 1, :, :] *= scale_h

        # flow = flow[0].cpu().permute(1, 2, 0).numpy()

        return flow