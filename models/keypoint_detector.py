import torch
import torch.nn as nn
from .utils import BNReluConv, upsample
from data.keypoint_utils import get_keypoints_from_heatmap
from itertools import chain


class KeypointDetector(nn.Module):
    def __init__(self, backbone, decoder, decoder_channels, num_keypoints):
        super(KeypointDetector, self).__init__()
        self.backbone = backbone
        self.decoder = decoder
        self.final_conv = BNReluConv(decoder_channels, num_keypoints, 1)

    def forward(self, x):
        B, C, H, W = x.shape
        x = self.backbone(x)
        x = self.decoder(x)
        x = self.final_conv(x)
        x = upsample(x, (H, W))
        keypoints = get_keypoints_from_heatmap(x)
        ret = {
            'keypoints': keypoints,
            'heatmap': x
        }
        return ret

    def forward_ms(self, x, scales):
        output_heatmaps = []
        B, C, H, W = x.shape
        for i, scale in enumerate(scales):
            x_scaled = upsample(x, (int(H * scale), int(W * scale)))
            out = self.forward(x_scaled)
            heatmap = upsample(out['heatmap'], (H, W))
            output_heatmaps.append(heatmap)
        mean_heatmap = torch.mean(torch.stack(output_heatmaps), dim=0)
        keypoints = get_keypoints_from_heatmap(mean_heatmap)
        ret = {
            'keypoints': keypoints,
            'heatmap': mean_heatmap
        }
        return ret

    def fine_tune_params(self):
        return self.backbone.parameters()

    def train_params(self):
        return chain(self.decoder.parameters(), self.final_conv.parameters())