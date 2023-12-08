import torch
import torch.nn.functional as F
import warnings

from torch import nn as nn

upsample = lambda x, size: F.interpolate(x, size, mode='bilinear', align_corners=False)
batchnorm_momentum = 0.01 / 2


def get_n_params(parameters):
    pp = 0
    for p in parameters:
        nn = 1
        for s in list(p.size()):
            nn = nn * s
        pp += nn
    return pp


class BNReluConv(nn.Sequential):
    def __init__(self, num_maps_in, num_maps_out, k=3, batch_norm=True, bn_momentum=0.1, bias=False, dilation=1):
        super(BNReluConv, self).__init__()
        if batch_norm:
            self.add_module('norm', nn.BatchNorm2d(num_maps_in, momentum=bn_momentum))
        self.add_module('relu', nn.ReLU(inplace=batch_norm is True))
        padding = k // 2
        self.add_module('conv', nn.Conv2d(num_maps_in, num_maps_out, kernel_size=k, padding=padding, bias=bias, dilation=dilation))


class Upsample(nn.Module):
    def __init__(
        self,
        num_maps_in,
        skip_maps_in,
        num_maps_out,
        k=3,
        use_skip=True,
    ):
        super(Upsample, self).__init__()
        self.bottleneck = BNReluConv(skip_maps_in, num_maps_in, k=1, batch_norm=True)
        self.blend_conv = BNReluConv(num_maps_in, num_maps_out, k=k, batch_norm=True)
        self.use_skip = use_skip
        self.upsampling_method = upsample

    def forward(self, x, skip):
        skip = self.bottleneck.forward(skip)
        skip_size = skip.size()[2:4]
        x = self.upsampling_method(x, skip_size)
        if self.use_skip:
            x = x + skip
        x = self.blend_conv.forward(x)
        return x


class SpatialPyramidPooling(nn.Module):
    def __init__(
        self,
        num_maps_in,
        bottleneck_size=512,
        level_size=128,
        out_size=128,
        grids=(6, 3, 2, 1),
    ):
        super(SpatialPyramidPooling, self).__init__()
        self.grids = grids
        self.upsampling_method = upsample
        self.bottleneck = BNReluConv(num_maps_in, bottleneck_size, k=1)

        num_features = bottleneck_size
        num_levels = len(grids)
        final_size = bottleneck_size + num_levels * level_size
        self.post_pool_convs = nn.ModuleList()
        for i in range(num_levels):
            self.post_pool_convs.append(BNReluConv(num_features, level_size, k=1))

        self.fuse = BNReluConv(final_size, out_size, k=1)

    def forward(self, x):
        levels = []
        target_size = x.shape[2:4]

        ar = target_size[1] / target_size[0]

        x = self.bottleneck(x)
        levels.append(x)

        for i, grid_size in enumerate(self.grids):
            grid = (grid_size, max(1, round(ar * grid_size)))
            x_pooled = F.adaptive_avg_pool2d(x, grid)
            level = self.post_pool_convs[i](x_pooled)
            level = self.upsampling_method(level, target_size)
            levels.append(level)
        x = torch.cat(levels, 1)
        x = self.fuse(x)
        return x


class Identity(nn.Module):
    def __init__(self, *args, **kwargs):
        super(Identity, self).__init__()

    def forward(self, input):
        return input