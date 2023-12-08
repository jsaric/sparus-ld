from models.resnet import resnet18, resnet34, resnet50, ResNet18_Weights, ResNet34_Weights, ResNet50_Weights
from models.swiftnet_decoder import SwiftNetDecoder
from models.keypoint_detector import KeypointDetector


def build_decoder(cfg):
    if cfg.MODEL.DECODER.TYPE == "SwiftNet":
        decoder = SwiftNetDecoder(
            in_channels=cfg.MODEL.DECODER.INPUT_CHANNELS,
            in_key=cfg.MODEL.DECODER.IN_KEY,
            skip_channels=cfg.MODEL.DECODER.SKIP_CHANNELS,
            skip_keys=cfg.MODEL.DECODER.SKIP_KEYS,
            decoder_channels=cfg.MODEL.DECODER.DECODER_CHANNELS,
            spp_grids=cfg.MODEL.DECODER.SWIFTNET_DECODDER.SPP_GRIDS,
            spp_bottleneck_size=cfg.MODEL.DECODER.SWIFTNET_DECODDER.SPP_BOTTLENECK_SIZE,
            spp_level_size=cfg.MODEL.DECODER.SWIFTNET_DECODDER.SPP_LEVEL_SIZE
        )
    else:
        raise NotImplementedError
    return decoder


def build_backbone(cfg):
    if cfg.MODEL.BACKBONE.TYPE == "ResNet":
        print("Building ResNet backbone...")
        if cfg.MODEL.BACKBONE.ResNet.DEPTH == 18:
            if cfg.MODEL.BACKBONE.ResNet.WEIGHTS == "IMAGENET1K_V1":
                print("Loading pretrained weights for ResNet18...")
                backbone = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
            else:
                backbone = resnet18(weights=None)
        elif cfg.MODEL.BACKBONE.ResNet.DEPTH == 34:
            if cfg.MODEL.BACKBONE.ResNet.WEIGHTS == "IMAGENET1K_V1":
                print("Loading pretrained weights for ResNet34...")
                backbone = resnet34(weights=ResNet34_Weights.IMAGENET1K_V1)
            else:
                backbone = resnet34(weights=None)
        elif cfg.MODEL.BACKBONE.ResNet.DEPTH == 50:
            if cfg.MODEL.BACKBONE.ResNet.WEIGHTS == "IMAGENET1K_V1":
                print("Loading pretrained weights for ResNet50...")
                backbone = resnet50(weights=ResNet50_Weights.IMAGENET1K_V1)
            else:
                backbone = resnet50(weights=None)
        else:
            raise NotImplementedError
    return backbone


def build_model(cfg):
    backbone = build_backbone(cfg)
    decoder = build_decoder(cfg)
    model = KeypointDetector(
        backbone=backbone,
        decoder=decoder,
        decoder_channels=cfg.MODEL.DECODER.DECODER_CHANNELS,
        num_keypoints=cfg.MODEL.NUM_KEYPOINTS
    )
    return model
