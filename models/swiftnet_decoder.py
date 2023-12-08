import torch
import torch.nn as nn
from .utils import SpatialPyramidPooling, Upsample


class SwiftNetDecoder(nn.Module):
    def __init__(
        self,
        in_channels,
        in_key,
        skip_channels,
        skip_keys,
        decoder_channels,
        spp_grids=(8, 4, 2, 1),
        spp_bottleneck_size=512,
        spp_level_size=128
    ):
        super(SwiftNetDecoder, self).__init__()

        self.spp = SpatialPyramidPooling(
            in_channels,
            bottleneck_size=spp_bottleneck_size,
            level_size=spp_level_size,
            out_size=decoder_channels,
            grids=spp_grids
        )

        self.feature_key = in_key
        self.decoder_stage = len(skip_channels)
        assert self.decoder_stage == len(skip_keys)
        self.low_level_key = skip_keys

        # Transform low-level feature
        upsamples = []
        # Top-down direction, i.e. starting from largest stride
        for i in range(self.decoder_stage):
            upsamples += [Upsample(decoder_channels, skip_channels[i], decoder_channels)]
        self.upsamples = nn.ModuleList(upsamples)

    def forward(self, features):
        x = features[self.feature_key]
        x = self.spp(x)

        for i in range(self.decoder_stage):
            k = self.low_level_key[i]
            iterable = isinstance(k, list) or isinstance(k, tuple)
            l = features[k] if not iterable else [features[ki] for ki in k]
            x = self.upsamples[i](x, l)
        return x