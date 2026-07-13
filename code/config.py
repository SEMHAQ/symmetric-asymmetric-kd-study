"""Configuration for KD experiments."""
import torch

# ===== Dataset =====
DATASET = 'cifar100'
NUM_CLASSES = 100
DATA_ROOT = '../data'

# ===== Training =====
EPOCHS = 60
BATCH_SIZE = 128
LEARNING_RATE = 0.05
MOMENTUM = 0.9
WEIGHT_DECAY = 5e-4
LR_SCHEDULER = 'cosine'  # cosine annealing
WARMUP_EPOCHS = 5

# ===== KD Hyperparameters =====
# KD (Hinton)
KD_TEMPERATURE = 4.0
KD_ALPHA = 0.5

# FitNet
FITNET_HINT_LAYER_T = 2  # layer index for hint (teacher)
FITNET_HINT_LAYER_S = 1  # layer index for guided (student)
FITNET_BETA = 1000.0

# AT
AT_BETA = 500.0

# SP
SP_BETA = 3000.0

# CC
CC_BETA = 0.02
CC_GAMMA = 0.4

# SACD (Symmetric-Asymmetric Collaborative Distillation)
SACD_LAMBDA = 1.0  # overall KD weight scale (cosine decay applied in training)

# ===== Hardware =====
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
NUM_WORKERS = 4
PIN_MEMORY = True

# ===== Teacher =====
TEACHER_ARCH = 'resnet34'
TEACHER_CKPT = '../experiments/checkpoints/teacher_resnet34_best.pth'

# ===== Student =====
STUDENT_ARCH = 'resnet18'
STUDENT_BASELINE_CKPT = '../experiments/checkpoints/student_resnet18_baseline.pth'
