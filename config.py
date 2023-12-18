from yacs.config import CfgNode as CN

_C = CN()
_C.INPUT = CN()
_C.DATASETS = CN()
_C.DATASETS.TRAIN = "SPAUR_1152x768"
_C.DATASETS.TRAIN_SPLIT = "train"
_C.DATASETS.VAL = "SPAUR_1152x768"
_C.DATASETS.VAL_SPLIT = "val"

_C.INPUT.SIZE = (768, 1152)
_C.INPUT.NORMALIZE = True
_C.INPUT.MEAN = (0.485, 0.456, 0.406)
_C.INPUT.STD = (0.229, 0.224, 0.225)
_C.INPUT.CROP_TYPE = "center"
_C.INPUT.RANDOM_ERASE = CN()
_C.INPUT.RANDOM_ERASE.ENABLED = False
_C.INPUT.RANDOM_ERASE.PROBABILITY = 0.1
_C.INPUT.RANDOM_ERASE.SCALE = (0.01, 0.1)
_C.INPUT.RANDOM_ERASE.RATIO = (0.7, 1.3)
_C.INPUT.RANDOM_ERASE.VALUE = (0, 0, 0)
_C.INPUT.RANDOM_ROTATION = CN()
_C.INPUT.RANDOM_ROTATION.ENABLED = True
_C.INPUT.RANDOM_ROTATION.ANGLE = 3
_C.INPUT.RANDOM_SHORTER_SIDE = CN()
_C.INPUT.RANDOM_SHORTER_SIDE.ENABLED = True
_C.INPUT.RANDOM_SHORTER_SIDE.MIN_SIZE = (700, 780)
_C.INPUT.RANDOM_SHORTER_SIDE.MAX_SIZE = 1200
_C.INPUT.COLOR_JITTER = CN()
_C.INPUT.COLOR_JITTER.ENABLED = True
_C.INPUT.COLOR_JITTER.BRIGHTNESS = (0.5, 1.3)
_C.INPUT.COLOR_JITTER.CONTRAST = (0.0, 0.0)
_C.INPUT.COLOR_JITTER.SATURATION = (0.5, 1.5)
_C.INPUT.COLOR_JITTER.HUE = (-0.1, 0.1)

_C.OPTIMIZER = CN()
_C.OPTIMIZER.TYPE = 'adam'
_C.OPTIMIZER.LR = 0.001
_C.OPTIMIZER.FINE_TUNE_LR_MULTIPLIER = 0.1
_C.OPTIMIZER.WEIGHT_DECAY = 0.0001
_C.OPTIMIZER.MOMENTUM = 0.9
_C.OPTIMIZER.SCHEDULER = CN()
_C.OPTIMIZER.SCHEDULER.TYPE = 'PolyLR'
_C.OPTIMIZER.SCHEDULER.STEP_INTERVAL = 1

_C.OPTIMIZER.SCHEDULER.POLYLR = CN()
_C.OPTIMIZER.SCHEDULER.POLYLR.POWER = 0.9
_C.OPTIMIZER.SCHEDULER.POLYLR.MAX_ITER = 30000

_C.OPTIMIZER.SCHEDULER.EXPLR = CN()
_C.OPTIMIZER.SCHEDULER.EXPLR.GAMMA = 0.98

_C.LOSS = CN()
_C.LOSS.TYPE = 'mse'
_C.LOSS.SIGMA = 4.0
_C.LOSS.TOP_K_PERCENT_PIXELS = 1.0

_C.TRAINER = CN()
_C.TRAINER.BATCH_SIZE = 8
_C.TRAINER.ITERATIONS = 30000
_C.TRAINER.EVAL_INTERVAL = 5000
_C.TRAINER.CHECKPOINT_INTERVAL = 5000
_C.TRAINER.NUM_WORKERS = 4
_C.TRAINER.LOG_INTERVAL = 100
_C.TRAINER.MS_EVAL = False
_C.TRAINER.MS_SCALES = (0.5, 0.75, 1.0, 1.25, 1.5, 2.0)

_C.MODEL = CN()
_C.MODEL.BACKBONE = CN()
_C.MODEL.BACKBONE.TYPE = 'ResNet'
_C.MODEL.BACKBONE.ResNet = CN()
_C.MODEL.BACKBONE.ResNet.DEPTH = 18
_C.MODEL.BACKBONE.ResNet.WEIGHTS = "IMAGENET1K_V1"


_C.MODEL.NUM_KEYPOINTS = 18
_C.MODEL.DECODER = CN()
_C.MODEL.DECODER.TYPE = 'SwiftNet'
_C.MODEL.DECODER.INPUT_CHANNELS = 512
_C.MODEL.DECODER.DECODER_CHANNELS = 128
_C.MODEL.DECODER.USE_SKIP = True
_C.MODEL.DECODER.IN_KEY = "res5"
_C.MODEL.DECODER.SKIP_KEYS = ["res4", "res3", "res2"]
_C.MODEL.DECODER.SKIP_CHANNELS = [256, 128, 64]
_C.MODEL.DECODER.SWIFTNET_DECODDER = CN()
_C.MODEL.DECODER.SWIFTNET_DECODDER.SPP_GRIDS = (8, 4, 2, 1)
_C.MODEL.DECODER.SWIFTNET_DECODDER.SPP_BOTTLENECK_SIZE = 512
_C.MODEL.DECODER.SWIFTNET_DECODDER.SPP_LEVEL_SIZE = 128

_C.EVALUATION = CN()
_C.EVALUATION.EVALUATORS = ("euclidean_distance_evaluator", "visualization_evaluator", "keypoint_similarity_evaluator", "tps_file_saver")
_C.EVALUATION.TPS_SAVER_SCALING_FACTOR = 1.0



_C.OUTPUT_DIR = 'output/testing/'

def get_cfg_defaults():
    return _C.clone()