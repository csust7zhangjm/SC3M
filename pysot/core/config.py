# Copyright (c) SenseTime. All Rights Reserved.

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

from yacs.config import CfgNode as CN

__C = CN()

cfg = __C

__C.META_ARC = "siamgat_googlenet"

__C.CUDA = True

# ------------------------------------------------------------------------ #
# Training options
# ------------------------------------------------------------------------ #
__C.TRAIN = CN()

__C.TRAIN.EXEMPLAR_SIZE = 127

__C.TRAIN.SEARCH_SIZE = 287

__C.TRAIN.OUTPUT_SIZE = 25

__C.TRAIN.RESUME = ''

__C.TRAIN.PRETRAINED = ''

__C.TRAIN.LOG_DIR = './logs'

__C.TRAIN.SNAPSHOT_DIR = './pth_fused_dpw'

__C.TRAIN.EPOCH = 20

__C.TRAIN.START_EPOCH = 0

__C.TRAIN.BATCH_SIZE = 32

# 测试NUM_WORKERS的大小
__C.TRAIN.NUM_WORKERS = 4  # 1 2 3 4

__C.TRAIN.MOMENTUM = 0.9

__C.TRAIN.WEIGHT_DECAY = 0.0001

__C.TRAIN.CLS_WEIGHT = 1.0

__C.TRAIN.LOC_WEIGHT = 3.0

__C.TRAIN.CEN_WEIGHT = 1.0

__C.TRAIN.PRINT_FREQ = 20

__C.TRAIN.LOG_GRADS = False

__C.TRAIN.GRAD_CLIP = 10.0

__C.TRAIN.BASE_LR = 0.005

__C.TRAIN.LR = CN()

__C.TRAIN.LR.TYPE = 'log'

__C.TRAIN.LR.KWARGS = CN(new_allowed=True)

__C.TRAIN.LR_WARMUP = CN()

__C.TRAIN.LR_WARMUP.WARMUP = True

__C.TRAIN.LR_WARMUP.TYPE = 'step'

__C.TRAIN.LR_WARMUP.EPOCH = 5

__C.TRAIN.LR_WARMUP.KWARGS = CN(new_allowed=True)

__C.TRAIN.NUM_CLASSES = 2

__C.TRAIN.NUM_CONVS = 4

__C.TRAIN.PRIOR_PROB = 0.01

__C.TRAIN.LOSS_ALPHA = 0.25

__C.TRAIN.LOSS_GAMMA = 2.0

# ------------------------------------------------------------------------ #
# Dataset options
# ------------------------------------------------------------------------ #
__C.DATASET = CN(new_allowed=True)

# Augmentation
# for template
__C.DATASET.TEMPLATE = CN()

# for detail discussion
__C.DATASET.TEMPLATE.SHIFT = 4

__C.DATASET.TEMPLATE.SCALE = 0.05

__C.DATASET.TEMPLATE.BLUR = 0.0

__C.DATASET.TEMPLATE.FLIP = 0.0

__C.DATASET.TEMPLATE.COLOR = 1.0

__C.DATASET.SEARCH = CN()

__C.DATASET.SEARCH.SHIFT = 64

__C.DATASET.SEARCH.SCALE = 0.18

__C.DATASET.SEARCH.BLUR = 0.0

__C.DATASET.SEARCH.FLIP = 0.0

__C.DATASET.SEARCH.COLOR = 1.0

# for detail discussion
__C.DATASET.NEG = 0.0

__C.DATASET.GRAY = 0.0

__C.DATASET.NAMES = ('VID', 'COCO', 'DET', 'YOUTUBEBB', 'GOT', 'LaSOT', 'TrackingNet')

__C.DATASET.VID = CN()
__C.DATASET.VID.ROOT = '/mnt/data/training_dataset/vid/crop511'
__C.DATASET.VID.ANNO = 'training_dataset/vid/train.json'
__C.DATASET.VID.FRAME_RANGE = 100
__C.DATASET.VID.NUM_USE = 100000  # repeat until reach NUM_USE

__C.DATASET.YOUTUBEBB = CN()
__C.DATASET.YOUTUBEBB.ROOT = '/mnt/data/ytbb/crop511'
__C.DATASET.YOUTUBEBB.ANNO = 'ytbb/train.json'
__C.DATASET.YOUTUBEBB.FRAME_RANGE = 3
__C.DATASET.YOUTUBEBB.NUM_USE = 200000

__C.DATASET.COCO = CN()
__C.DATASET.COCO.ROOT = '/mnt/data/coco/crop511'
__C.DATASET.COCO.ANNO = 'coco/train2017.json'
__C.DATASET.COCO.FRAME_RANGE = 1
__C.DATASET.COCO.NUM_USE = 100000

__C.DATASET.DET = CN()
__C.DATASET.DET.ROOT = '/mnt/data/training_dataset/det/crop511'
__C.DATASET.DET.ANNO = 'training_dataset/det/train.json'
__C.DATASET.DET.FRAME_RANGE = 1
__C.DATASET.DET.NUM_USE = 200000

__C.DATASET.GOT = CN()
__C.DATASET.GOT.ROOT = '/mnt/data/training_dataset/got10k/crop511'
__C.DATASET.GOT.ANNO = 'training_dataset/got10k/train.json'
__C.DATASET.GOT.FRAME_RANGE = 100
__C.DATASET.GOT.NUM_USE = 200000

__C.DATASET.LaSOT = CN()
__C.DATASET.LaSOT.ROOT = '/PATH/TO/LaSOT'
__C.DATASET.LaSOT.ANNO = 'training_dataset/lasot/train.json'
__C.DATASET.LaSOT.FRAME_RANGE = 100
__C.DATASET.LaSOT.NUM_USE = 200000

__C.DATASET.TrackingNet = CN()
__C.DATASET.TrackingNet.ROOT = '/PATH/TO/TrackingNet'
__C.DATASET.TrackingNet.ANNO = 'training_dataset/trackingnet/train.json'
__C.DATASET.TrackingNet.FRAME_RANGE = 100
__C.DATASET.TrackingNet.NUM_USE = 200000

__C.DATASET.VIDEOS_PER_EPOCH = 800000

# ------------------------------------------------------------------------ #
# Backbone options
# ------------------------------------------------------------------------ #
__C.BACKBONE = CN()

# Backbone type, current only support googlenet;alexnet;
__C.BACKBONE.TYPE = 'googlenet'

__C.BACKBONE.KWARGS = CN(new_allowed=True)

# Pretrained backbone weights
__C.BACKBONE.PRETRAINED = ''

# Train backbone layers
__C.BACKBONE.TRAIN_LAYERS = []

# Train channel_layer
__C.BACKBONE.CHANNEL_REDUCE_LAYERS = []

# Layer LR
__C.BACKBONE.LAYERS_LR = 0.1

# Crop_pad
__C.BACKBONE.CROP_PAD = 4

# Switch to train layer
__C.BACKBONE.TRAIN_EPOCH = 10

# Backbone offset
__C.BACKBONE.OFFSET = 13

# Backbone stride
__C.BACKBONE.STRIDE = 8

# ------------------------------------------------------------------------ #
# Tracker options
# ------------------------------------------------------------------------ #
__C.TRACK = CN()

# SiamGAT
__C.TRAIN.ATTENTION = True

__C.TRACK.TYPE = 'SiamGATTracker'

# Scale penalty
__C.TRACK.PENALTY_K = 0.04

# Window influence
__C.TRACK.WINDOW_INFLUENCE = 0.44

# Interpolation learning rate
__C.TRACK.LR = 0.4

# Exemplar size
__C.TRACK.EXEMPLAR_SIZE = 127

# Instance size
__C.TRACK.INSTANCE_SIZE = 287

# Context amount
__C.TRACK.CONTEXT_AMOUNT = 0.5

__C.TRACK.STRIDE = 8

__C.TRACK.OFFSET = 45

__C.TRACK.SCORE_SIZE = 25

__C.TRACK.hanming = True

__C.TRACK.REGION_S = 0.1

__C.TRACK.REGION_L = 0.44

# ------------------------------------------------------------------------ #
# HP_SEARCH parameters
# ------------------------------------------------------------------------ #
__C.HP_SEARCH = CN()

# lr pk wi
__C.HP_SEARCH.OTB100 =  [0.28, 0.16, 0.4]  #[0.35, 0.20091, 0.45]# [0.32, 0.3, 0.38]#[0.4, 0.04, 0.44]# [0.28, 0.16, 0.4] [0.35, 0.2, 0.45]

__C.HP_SEARCH.GOT_10k = [0.8, 0.01, 0.1] # [0.7, 0.06, 0.1][0.9, 0.25, 0.35]

__C.HP_SEARCH.UAV123 = [0.24, 0.04, 0.04]  #[0.4, 0.04, 0.440] [0.24, 0.04, 0.04] #[0.4, 0.2, 0.3] #[0.24, 0.04, 0.04]

__C.HP_SEARCH.LaSOT = [0.3, 0.05, 0.18] # 测不了
#_pk-0.120_wi-0.450_lr-0.320

# TODO : 测试超参数
__C.HP_SEARCH.VOT2016 =[0.4,0.03999,0.44]   # [0.320, 0.120, 0.450] #[0.30, 0.14, 0.45] [0.295, 0.055, 0.42]

__C.HP_SEARCH.VOT2018 =[0.4,0.03999,0.44]   #[0.32, 0.20091, 0.48] #[0.295, 0.055, 0.42]  #[0.320, 0.160, 0.450] #[0.330, 0.04,0.33] [0.320, 0.120, 0.450]   [0.7, 0.02, 0.35] [0.4, 0.04, 0.44] [0.295, 0.055, 0.42] [0.30, 0.14, 0.45]
# ------------------------------------------------------------------------ #
# mask options
# -------------------------- --------------------------------------------- #
__C.MASK = CN()

# Whether to use mask generate segmentation
__C.MASK.MASK = False

# Mask type
__C.MASK.TYPE = "MaskCorr"

__C.MASK.KWARGS = CN(new_allowed=True)

__C.REFINE = CN()

# Mask refine
__C.REFINE.REFINE = False

# Refine type
__C.REFINE.TYPE = "Refine"

# Mask threshold
__C.TRACK.MASK_THERSHOLD = 0.30

# Mask output size
__C.TRACK.MASK_OUTPUT_SIZE = 127
