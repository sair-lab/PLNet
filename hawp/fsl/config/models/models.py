from yacs.config import CfgNode as CN
from .shg import HGNETS
from .resnets import RESNETS
from .head import PARSING_HEAD

MODELS = CN()

MODELS.NAME = "Hourglass"
MODELS.HGNETS = HGNETS
MODELS.RESNETS = RESNETS
MODELS.DEVICE = "cuda"
MODELS.WEIGHTS = ""
MODELS.HEAD_SIZE  = [[3], [1], [1], [2], [2]] 
MODELS.OUT_FEATURE_CHANNELS = 256

MODELS.LOSS_WEIGHTS = CN(new_allowed=True)
MODELS.PARSING_HEAD   = PARSING_HEAD
MODELS.SCALE = 1.0

MODELS.USE_LINE_HEATMAP = False
MODELS.USE_HR_JUNCTION = False

MODELS.LOI_POOLING = CN()
MODELS.LOI_POOLING.USE_INIT_LINES = True
MODELS.LOI_POOLING.NUM_POINTS = 32
MODELS.LOI_POOLING.DIM_EDGE_FEATURE = 16
MODELS.LOI_POOLING.DIM_JUNCTION_FEATURE = 128
MODELS.LOI_POOLING.DIM_FC = 1024
MODELS.LOI_POOLING.TYPE = 'softmax'
MODELS.LOI_POOLING.LAYER_NORM = False
MODELS.LOI_POOLING.ACTIVATION = 'relu'

MODELS.FOCAL_LOSS = CN()
MODELS.FOCAL_LOSS.ALPHA = -1.0
MODELS.FOCAL_LOSS.GAMMA = 0.0