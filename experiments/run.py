import os
import sys
import json
import time
import datetime
import logging
import warnings
import random

import pandas as pd
import numpy as np
from tqdm import tqdm

import wandb
import optuna

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from torchinfo import summary
from torch.utils.data import DataLoader, TensorDataset, Dataset, SubsetRandomSampler
from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix
from gradcam import GradCAM
import matplotlib.pyplot as plt

import model as md
import config as CONFIG
import utils

warnings.filterwarnings("ignore")
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

os.environ["WANDB_API_KEY"] = "fd827dd887a26e0bcb19da142798f87c9da33cb6"
os.environ["WANDB_MODE"] = "offline"

# sweep_config_cnn_32x16 = {
#     "method": "grid",
#     "name": "cnn_32x16_y_dropout_y_rules",
#     "metric": {"goal": "maximize", "name": "Recall"},
#     "parameters": {
#         "conv1_kernel_size": {"values": [9]},
#         "conv2_kernel_size": {"values": [5]},
#         "conv3_kernel_size": {"values": [3]},
#         "conv4_kernel_size": {"values": [3]},
#         "learning_rate": {"values": [0.0001]},
#         "dropout": {"values": [0.2]},
#         "conv1_out_channels": {"values": [32]},
#         "conv2_out_channels": {"values": [64]},
#         "conv3_out_channels": {"values": [128]},
#         "conv4_out_channels": {"values": [512]},
#         "row_th": {"values": [15]},
#         "col_th": {"values": [7]},
#         "num_epochs": {"values": [100]},
#         "repeat_times": {"values": [i for i in range(3)] },
#     }
# }
# sweep_config_cnn_64x32 = {
#     "method": "grid",
#     "name": "cnn_64x32_y_dropout",
#     "metric": {"goal": "maximize", "name": "Recall"},
#     "parameters": {
#         "conv1_kernel_size": {"values": [5, 7, 9, 11, 13, 15, 17]},
#         "conv2_kernel_size": {"values": [3, 5, 7, 9]},
#         "conv3_kernel_size": {"values": [3, 5]},
#         "conv4_kernel_size": {"values": [3]},
#         # "conv1_kernel_size": {"values": [17]},
#         # "conv2_kernel_size": {"values": [9]},
#         # "conv3_kernel_size": {"values": [5]},
#         # "conv4_kernel_size": {"values": [3]},
#         "learning_rate": {"values": [0.001]},
#         "dropout": {"values": [0.1]},
#         "conv1_out_channels": {"values": [32]},
#         "conv2_out_channels": {"values": [64]},
#         "conv3_out_channels": {"values": [128]},
#         "conv4_out_channels": {"values": [256]}
#     }
# }
# sweep_config_resnet = {
#     "method": "grid",
#     "name": "resnet_32x16_y_dropout_y_rules",
#     "metric": {"goal": "maximize", "name": "Recall"},
#     "parameters": {
#         "conv1_kernel_size": {"values": [9]},
#         "conv2_kernel_size": {"values": [5]},
#         "conv3_kernel_size": {"values": [3]},
#         "conv4_kernel_size": {"values": [3]},
#         "learning_rate": {"values": [0.0002]},
#         "dropout": {"values": [0.2]},
#         "conv1_out_channels": {"values": [32]},
#         "conv2_out_channels": {"values": [64]},
#         "conv3_out_channels": {"values": [128]},
#         "conv4_out_channels": {"values": [512]},
#         "row_th": {"values": [15]},
#         "col_th": {"values": [5]},
#         "num_epochs": {"values": [150]},
#         "repeat_times": {"values": [i for i in range(5)] },
#     }
# }
# sweep_config_vit = {
#     "method": "grid",
#     "name": "vit",
#     "metric": {"goal": "maximize", "name": "Recall"},
#     "parameters": {
#         "patch_size": {"values": [2, 4, 8]},
#         "hidden_dim": {"values": [8, 16, 32, 64]},
#         "num_heads": {"values": [4, 8, 16]},
#         "num_layers": {"values": [2, 4, 8]},
#         "learning_rate": {"values": [0.0001, 0.00001]}
#         # "patch_size": {"values": [1]},
#         # "hidden_dim": {"values": [8]},
#         # "num_heads": {"values": [4]},
#         # "num_layers": {"values": [2]},
#         # "learning_rate": {"values": [0.0001]}
#     }
# }




sweep_configs = {
    "cnn_32x16": {
        "method": "grid",
        "name": "cnn_32x16_y_dropout_y_rules",
        "metric": {"goal": "maximize", "name": "Recall"},
        "parameters": {
            "conv1_kernel_size": {"values": [9]},
            "conv2_kernel_size": {"values": [5]},
            "conv3_kernel_size": {"values": [3]},
            "conv4_kernel_size": {"values": [3]},
            "learning_rate": {"values": [0.0001]},
            "dropout": {"values": [0.2]},
            "conv1_out_channels": {"values": [32]},
            "conv2_out_channels": {"values": [64]},
            "conv3_out_channels": {"values": [128]},
            "conv4_out_channels": {"values": [512]},
            "row_th": {"values": [15]},
            "col_th": {"values": [7]},
            "num_epochs": {"values": [100]},
            "repeat_times": {"values": [i for i in range(3)] },
        }
    },
    "cnn_64x32": {
        "method": "grid",
        "name": "cnn_64x32_y_dropout",
        "metric": {"goal": "maximize", "name": "Recall"},
        "parameters": {
            "conv1_kernel_size": {"values": [5, 7, 9, 11, 13, 15, 17]},
            "conv2_kernel_size": {"values": [3, 5, 7, 9]},
            "conv3_kernel_size": {"values": [3, 5]},
            "conv4_kernel_size": {"values": [3]},
            # "conv1_kernel_size": {"values": [17]},
            # "conv2_kernel_size": {"values": [9]},
            # "conv3_kernel_size": {"values": [5]},
            # "conv4_kernel_size": {"values": [3]},
            "learning_rate": {"values": [0.001]},
            "dropout": {"values": [0.1]},
            "conv1_out_channels": {"values": [32]},
            "conv2_out_channels": {"values": [64]},
            "conv3_out_channels": {"values": [128]},
            "conv4_out_channels": {"values": [256]}
        }
    },
    "resnet": {
        "method": "grid",
        "name": "resnet_32x16_y_dropout_y_rules",
        "metric": {"goal": "maximize", "name": "Recall"},
        "parameters": {
            "conv1_kernel_size": {"values": [9]},
            "conv2_kernel_size": {"values": [5]},
            "conv3_kernel_size": {"values": [3]},
            "conv4_kernel_size": {"values": [3]},
            "learning_rate": {"values": [0.0002]},
            "dropout": {"values": [0.2]},
            "conv1_out_channels": {"values": [32]},
            "conv2_out_channels": {"values": [64]},
            "conv3_out_channels": {"values": [128]},
            "conv4_out_channels": {"values": [512]},
            "row_th": {"values": [15]},
            "col_th": {"values": [5]},
            "num_epochs": {"values": [150]},
            "repeat_times": {"values": [i for i in range(1)] },
        }
    },
    "resnet_multichannels": {
        "method": "grid",
        "name": "resnet_32x16_y_dropout_y_rules_multichannels",
        "metric": {"goal": "maximize", "name": "Recall"},
        "parameters": {
            "conv1_kernel_size": {"values": [9]},
            "conv2_kernel_size": {"values": [5]},
            "conv3_kernel_size": {"values": [3]},
            "conv4_kernel_size": {"values": [3]},
            "learning_rate": {"values": [0.0001, 0.0002, 0.001]},
            "dropout": {"values": [0.2]},
            "conv1_out_channels": {"values": [32]},
            "conv2_out_channels": {"values": [64]},
            "conv3_out_channels": {"values": [128]},
            "conv4_out_channels": {"values": [512]},
            "row_th": {"values": [15]},
            "col_th": {"values": [5]},
            "num_epochs": {"values": [100]},
            "repeat_times": {"values": [i for i in range(3)] },
        }
    },
    "vit": {
        "method": "grid",
        "name": "vit",
        "metric": {"goal": "maximize", "name": "Recall"},
        "parameters": {
            "patch_size": {"values": [2, 4, 8]},
            "hidden_dim": {"values": [8, 16, 32, 64]},
            "num_heads": {"values": [4, 8, 16]},
            "num_layers": {"values": [2, 4, 8]},
            "learning_rate": {"values": [0.0001, 0.00001]}
            # "patch_size": {"values": [1]},
            # "hidden_dim": {"values": [8]},
            # "num_heads": {"values": [4]},
            # "num_layers": {"values": [2]},
            # "learning_rate": {"values": [0.0001]}
        }
    },
    "exp_cnn_32x16_l2_c1_worule": {
        "method": "grid",
        "name": "exp_cnn_32x16_l2_c1_worule",
        "metric": {"goal": "maximize", "name": "Recall"},
        "parameters": {
            "conv1_kernel_size": {"values": [9, 7, 5, 3]},
            "conv2_kernel_size": {"values": [7, 5, 3]},
            "conv3_kernel_size": {"values": [5, 3]},
            "conv4_kernel_size": {"values": [3]},
            "learning_rate": {"values": [0.0001]},
            "channel": {"values": [1]},
            "dropout": {"values": [0.2]},
            "conv1_out_channels": {"values": [64]},
            "conv2_out_channels": {"values": [128]},
            "row_th": {"values": [15]},
            "col_th": {"values": [7]},
            "num_epochs": {"values": [150]},
            "repeat_times": {"values": [i for i in range(1)] },
        }
    },
    "exp_cnn_32x16_l2_c1_wrule": {
        "method": "grid",
        "name": "exp_cnn_32x16_l2_c1_wrule",
        "metric": {"goal": "maximize", "name": "Recall"},
        "parameters": {
            "conv1_kernel_size": {"values": [9]},
            "conv2_kernel_size": {"values": [5]},
            "conv3_kernel_size": {"values": [3]},
            "conv4_kernel_size": {"values": [3]},
            "learning_rate": {"values": [0.0001]},
            "channel": {"values": [1]},
            "dropout": {"values": [0.2]},
            "conv1_out_channels": {"values": [64]},
            "conv2_out_channels": {"values": [128]},
            "row_th": {"values": [15]},
            "col_th": {"values": [7]},
            "num_epochs": {"values": [150]},
            "repeat_times": {"values": [i for i in range(10)] },
        }
    },
    "exp_cnn_32x16_l2_c3_worule": {
        "method": "grid",
        "name": "exp_cnn_32x16_l2_c3_worule",
        "metric": {"goal": "maximize", "name": "Recall"},
        "parameters": {
            "conv1_kernel_size": {"values": [9, 7, 5, 3]},
            "conv2_kernel_size": {"values": [7, 5, 3]},
            "conv3_kernel_size": {"values": [5, 3]},
            "conv4_kernel_size": {"values": [3]},
            "learning_rate": {"values": [0.0001, 0.00001]},
            "channel": {"values": [3]},
            "dropout": {"values": [0.1]},
            "conv1_out_channels": {"values": [64]},
            "conv2_out_channels": {"values": [128]},
            # "row_th": {"values": [0, 3.2, 6.4, 9.6, 12.8, 16, 19.2, 22.4, 25.6, 32]},
            # "col_th": {"values": [0.0, 1.6, 3.2, 4.8, 6.4, 8.0, 9.6, 11.2, 12.8, 16]},
            "row_th": {"values": [0]},
            "col_th": {"values": [0]},
            "num_epochs": {"values": [150]},
            "repeat_times": {"values": [i for i in range(1)] },
        }
    },
    "exp_cnn_32x16_l2_c3_wrule": {
        "method": "grid",
        "name": "exp_cnn_32x16_l2_c3_wrule",
        "metric": {"goal": "maximize", "name": "Recall"},
        "parameters": {
            "conv1_kernel_size": {"values": [9]},
            "conv2_kernel_size": {"values": [7]},
            "conv3_kernel_size": {"values": [3]},
            "conv4_kernel_size": {"values": [3]},
            "learning_rate": {"values": [0.0001]},
            "channel": {"values": [3]},
            "dropout": {"values": [0.2]},
            "conv1_out_channels": {"values": [64]},
            "conv2_out_channels": {"values": [128]},
            "row_th": {"values": [0, 3.2, 6.4, 9.6, 12.8, 16, 19.2, 22.4, 25.6, 32]},
            "col_th": {"values": [0.0, 1.6, 3.2, 4.8, 6.4, 8.0, 9.6, 11.2, 12.8, 16]},
            "num_epochs": {"values": [100]},
            "repeat_times": {"values": [i for i in range(1)] },
        }
    },
    "exp_cnn_32x16_l4_c1_worule": {
        "method": "grid",
        "name": "exp_cnn_32x16_l4_c1_worule",
        "metric": {"goal": "maximize", "name": "Recall"},
        "parameters": {
            "conv1_kernel_size": {"values": [9, 7, 5, 3]},
            "conv2_kernel_size": {"values": [7, 5, 3]},
            "conv3_kernel_size": {"values": [5, 3]},
            "conv4_kernel_size": {"values": [3]},
            "learning_rate": {"values": [0.0001]},
            "channel": {"values": [1]},
            "dropout": {"values": [0.2]},
            "conv1_out_channels": {"values": [32]},
            "conv2_out_channels": {"values": [64]},
            "conv3_out_channels": {"values": [128]},
            "conv4_out_channels": {"values": [512]},
            "row_th": {"values": [15]},
            "col_th": {"values": [7]},
            "num_epochs": {"values": [200]},
            "repeat_times": {"values": [i for i in range(1)] },
        }
    },
    "exp_cnn_32x16_l4_c1_wrule": {
        "method": "grid",
        "name": "exp_cnn_32x16_l4_c1_wrule",
        "metric": {"goal": "maximize", "name": "Recall"},
        "parameters": {
            "conv1_kernel_size": {"values": [9, 7, 5, 3]},
            "conv2_kernel_size": {"values": [7, 5, 3]},
            "conv3_kernel_size": {"values": [5, 3]},
            "conv4_kernel_size": {"values": [3]},
            "learning_rate": {"values": [0.0001]},
            "channel": {"values": [1]},
            "dropout": {"values": [0.2]},
            "conv1_out_channels": {"values": [32]},
            "conv2_out_channels": {"values": [64]},
            "conv3_out_channels": {"values": [128]},
            "conv4_out_channels": {"values": [512]},
            "row_th": {"values": [15]},
            "col_th": {"values": [7]},
            "num_epochs": {"values": [150]},
            "repeat_times": {"values": [i for i in range(10)] },
        }
    },
    "exp_cnn_32x16_l4_c3_worule": {
        "method": "grid",
        "name": "exp_cnn_32x16_l4_c3_worule",
        "metric": {"goal": "maximize", "name": "Recall"},
        "parameters": {
            "conv1_kernel_size": {"values": [9, 7, 5, 3]},
            "conv2_kernel_size": {"values": [7, 5, 3]},
            "conv3_kernel_size": {"values": [5, 3]},
            "conv4_kernel_size": {"values": [3]},
            "learning_rate": {"values": [0.0001, 0.00001]},
            "channel": {"values": [3]},
            "dropout": {"values": [0.2]},
            "conv1_out_channels": {"values": [32]},
            "conv2_out_channels": {"values": [64]},
            "conv3_out_channels": {"values": [128]},
            "conv4_out_channels": {"values": [512]},
            "row_th": {"values": [15]},
            "col_th": {"values": [7]},
            "num_epochs": {"values": [200]},
            "repeat_times": {"values": [i for i in range(1)] },
        }
    },
    "exp_cnn_32x16_l4_c3_wrule": {
        "method": "grid",
        "name": "exp_cnn_32x16_l4_c3_wrule",
        "metric": {"goal": "maximize", "name": "Recall"},
        "parameters": {
            "conv1_kernel_size": {"values": [9]},
            "conv2_kernel_size": {"values": [5]},
            "conv3_kernel_size": {"values": [3]},
            "conv4_kernel_size": {"values": [3]},
            "learning_rate": {"values": [0.0001]},
            "channel": {"values": [3]},
            "dropout": {"values": [0.2]},
            "conv1_out_channels": {"values": [32]},
            "conv2_out_channels": {"values": [64]},
            "conv3_out_channels": {"values": [128]},
            "conv4_out_channels": {"values": [512]},
            "row_th": {"values": [0, 3.2, 6.4, 9.6, 12.8, 16, 19.2, 22.4, 25.6, 32]},
            "col_th": {"values": [0.0, 1.6, 3.2, 4.8, 6.4, 8.0, 9.6, 11.2, 12.8, 16]},
            "num_epochs": {"values": [150]},
            "repeat_times": {"values": [i for i in range(1)] },
        }
    },
    "exp_resnet_32x16_l2_c1_worule": {
        "method": "grid",
        "name": "exp_resnet_32x16_l2_c1_worule",
        "metric": {"goal": "maximize", "name": "Recall"},
        "parameters": {
            "conv1_kernel_size": {"values": [9]},
            "conv2_kernel_size": {"values": [7]},
            "conv3_kernel_size": {"values": [3]},
            "conv4_kernel_size": {"values": [3]},
            "learning_rate": {"values": [0.0001]},
            "channel": {"values": [1]},
            "dropout": {"values": [0.2]},
            "conv1_out_channels": {"values": [32]},
            "conv2_out_channels": {"values": [64]},
            "conv3_out_channels": {"values": [128]},
            "conv4_out_channels": {"values": [512]},
            "row_th": {"values": [15]},
            "col_th": {"values": [7]},
            "num_epochs": {"values": [150]},
            "repeat_times": {"values": [i for i in range(10)] },
        }
    },
    "exp_resnet_32x16_l2_c1_wrule": {
        "method": "grid",
        "name": "exp_resnet_32x16_l2_c1_wrule",
        "metric": {"goal": "maximize", "name": "Recall"},
        "parameters": {
            "conv1_kernel_size": {"values": [9]},
            "conv2_kernel_size": {"values": [7]},
            "conv3_kernel_size": {"values": [3]},
            "conv4_kernel_size": {"values": [3]},
            "learning_rate": {"values": [0.0001]},
            "channel": {"values": [1]},
            "dropout": {"values": [0.2]},
            "conv1_out_channels": {"values": [32]},
            "conv2_out_channels": {"values": [64]},
            "conv3_out_channels": {"values": [128]},
            "conv4_out_channels": {"values": [512]},
            "row_th": {"values": [0, 3.2, 6.4, 9.6, 12.8, 16, 19.2, 22.4, 25.6, 32]},
            "col_th": {"values": [0.0, 1.6, 3.2, 4.8, 6.4, 8.0, 9.6, 11.2, 12.8, 16]},
            "num_epochs": {"values": [150]},
            "repeat_times": {"values": [i for i in range(1)] },
        }
    },
    "exp_resnet_32x16_l2_c3_worule": {
        "method": "grid",
        "name": "exp_resnet_32x16_l2_c3_worule",
        "metric": {"goal": "maximize", "name": "Recall"},
        "parameters": {
            "conv1_kernel_size": {"values": [9]},
            "conv2_kernel_size": {"values": [7]},
            "conv3_kernel_size": {"values": [3]},
            "conv4_kernel_size": {"values": [3]},
            "learning_rate": {"values": [0.0001]},
            "channel": {"values": [3]},
            "dropout": {"values": [0.2]},
            "conv1_out_channels": {"values": [32]},
            "conv2_out_channels": {"values": [64]},
            "conv3_out_channels": {"values": [128]},
            "conv4_out_channels": {"values": [512]},
            "row_th": {"values": [0, 3.2, 6.4, 9.6, 12.8, 16, 19.2, 22.4, 25.6, 32]},
            "col_th": {"values": [0.0, 1.6, 3.2, 4.8, 6.4, 8.0, 9.6, 11.2, 12.8, 16]},
            "num_epochs": {"values": [150]},
            "repeat_times": {"values": [i for i in range(10)] },
        }
    },
    "exp_resnet_32x16_l2_c3_wrule": {
        "method": "grid",
        "name": "exp_resnet_32x16_l2_c3_wrule",
        "metric": {"goal": "maximize", "name": "Recall"},
        "parameters": {
            "conv1_kernel_size": {"values": [9]},
            "conv2_kernel_size": {"values": [7]},
            "conv3_kernel_size": {"values": [3]},
            "conv4_kernel_size": {"values": [3]},
            "learning_rate": {"values": [0.0001]},
            "channel": {"values": [3]},
            "dropout": {"values": [0.2]},
            "conv1_out_channels": {"values": [32]},
            "conv2_out_channels": {"values": [64]},
            "conv3_out_channels": {"values": [128]},
            "conv4_out_channels": {"values": [512]},
            "row_th": {"values": [0, 3.2, 6.4, 9.6, 12.8, 16, 19.2, 22.4, 25.6, 32]},
            "col_th": {"values": [0.0, 1.6, 3.2, 4.8, 6.4, 8.0, 9.6, 11.2, 12.8, 16]},
            "num_epochs": {"values": [70]},
            "repeat_times": {"values": [i for i in range(1)] },
        }
    },
    "exp_resnet_32x16_l4_c1_worule": {
        "method": "grid",
        "name": "exp_resnet_32x16_l4_c1_worule",
        "metric": {"goal": "maximize", "name": "Recall"},
        "parameters": {
            "conv1_kernel_size": {"values": [9]},
            "conv2_kernel_size": {"values": [7]},
            "conv3_kernel_size": {"values": [3]},
            "conv4_kernel_size": {"values": [3]},
            "learning_rate": {"values": [0.0001]},
            "channel": {"values": [1]},
            "dropout": {"values": [0.2]},
            "conv1_out_channels": {"values": [32]},
            "conv2_out_channels": {"values": [64]},
            "conv3_out_channels": {"values": [128]},
            "conv4_out_channels": {"values": [512]},
            "row_th": {"values": [15]},
            "col_th": {"values": [7]},
            "num_epochs": {"values": [150]},
            "repeat_times": {"values": [i for i in range(10)] },
        }
    },
    "exp_resnet_32x16_l4_c1_wrule": {
        "method": "grid",
        "name": "exp_resnet_32x16_l4_c1_wrule",
        "metric": {"goal": "maximize", "name": "Recall"},
        "parameters": {
            "conv1_kernel_size": {"values": [9]},
            "conv2_kernel_size": {"values": [7]},
            "conv3_kernel_size": {"values": [3]},
            "conv4_kernel_size": {"values": [3]},
            "learning_rate": {"values": [0.0001]},
            "channel": {"values": [1]},
            "dropout": {"values": [0.2]},
            "conv1_out_channels": {"values": [32]},
            "conv2_out_channels": {"values": [64]},
            "conv3_out_channels": {"values": [128]},
            "conv4_out_channels": {"values": [512]},
            "row_th": {"values": [16]},
            "col_th": {"values": [8]},
            "num_epochs": {"values": [150]},
            "repeat_times": {"values": [i for i in range(1)] },
        }
    },
    "exp_resnet_32x16_l4_c3_worule": {
        "method": "grid",
        "name": "exp_resnet_32x16_l4_c3_worule",
        "metric": {"goal": "maximize", "name": "Recall"},
        "parameters": {
            "conv1_kernel_size": {"values": [9]},
            "conv2_kernel_size": {"values": [7]},
            "conv3_kernel_size": {"values": [3]},
            "conv4_kernel_size": {"values": [3]},
            "learning_rate": {"values": [0.0001]},
            "channel": {"values": [3]},
            "dropout": {"values": [0.0]},
            "conv1_out_channels": {"values": [32]},
            "conv2_out_channels": {"values": [64]},
            "conv3_out_channels": {"values": [128]},
            "conv4_out_channels": {"values": [512]},
            "row_th": {"values": [0, 3.2, 6.4, 9.6, 12.8, 16, 19.2, 22.4, 25.6, 32]},
            "col_th": {"values": [0.0, 1.6, 3.2, 4.8, 6.4, 8.0, 9.6, 11.2, 12.8, 16]},
            "num_epochs": {"values": [100]},
            "repeat_times": {"values": [i for i in range(1)] },
        }
    },
    "exp_resnet_32x16_l4_c3_wrule": {
        "method": "grid",
        "name": "exp_resnet_32x16_l4_c3_wrule",
        "metric": {"goal": "maximize", "name": "Recall"},
        "parameters": {
            "conv1_kernel_size": {"values": [9]},
            "conv2_kernel_size": {"values": [7]},
            "conv3_kernel_size": {"values": [3]},
            "conv4_kernel_size": {"values": [3]},
            "learning_rate": {"values": [0.0001]},
            "channel": {"values": [3]},
            "dropout": {"values": [0.2]},
            "conv1_out_channels": {"values": [32]},
            "conv2_out_channels": {"values": [64]},
            "conv3_out_channels": {"values": [128]},
            "conv4_out_channels": {"values": [512]},
            "row_th": {"values": [0, 3.2, 6.4, 9.6, 12.8, 16, 19.2, 22.4, 25.6, 32]},
            "col_th": {"values": [0.0, 1.6, 3.2, 4.8, 6.4, 8.0, 9.6, 11.2, 12.8, 16]},
            "num_epochs": {"values": [150]},
            "repeat_times": {"values": [i for i in range(1)] },
        }
    },
    "exp_vit_32x16_l2_c1_worule": {
        "method": "grid",
        "name": "exp_vit_32x16_l2_c1_worule",
        "metric": {"goal": "maximize", "name": "Recall"},
        "parameters": {
            "patch_size": {"values": [4]},
            "hidden_dim": {"values": [8]},
            "num_heads": {"values": [8]},
            "num_layers": {"values": [2]},
            "learning_rate": {"values": [0.0001]},
            "row_th": {"values": [15]},
            "col_th": {"values": [7]},
            "num_epochs": {"values": [150]},
            "channel": {"values": [1]},
            "repeat_times": {"values": [i for i in range(10)] },
        }
    },
    "exp_vit_32x16_l2_c1_wrule": {
        "method": "grid",
        "name": "exp_vit_32x16_l2_c1_wrule",
        "metric": {"goal": "maximize", "name": "Recall"},
        "parameters": {
            "patch_size": {"values": [4]},
            "hidden_dim": {"values": [8]},
            "num_heads": {"values": [8]},
            "num_layers": {"values": [2]},
            "learning_rate": {"values": [0.0001]},
            "row_th": {"values": [15]},
            "col_th": {"values": [7]},
            "channel": {"values": [1]},
            "num_epochs": {"values": [150]},
            "repeat_times": {"values": [i for i in range(10)] },
        }
    },
    "exp_vit_32x16_l2_c3_worule": {
        "method": "grid",
        "name": "exp_vit_32x16_l2_c3_worule",
        "metric": {"goal": "maximize", "name": "Recall"},
        "parameters": {
            "patch_size": {"values": [4]},
            "hidden_dim": {"values": [8]},
            "num_heads": {"values": [8]},
            "num_layers": {"values": [2]},
            "learning_rate": {"values": [0.0001]},
            "row_th": {"values": [15]},
            "col_th": {"values": [7]},
            "num_epochs": {"values": [150]},
            "channel": {"values": [3]},
            "repeat_times": {"values": [i for i in range(10)] },
        }
    },
    "exp_vit_32x16_l2_c3_wrule": {
        "method": "grid",
        "name": "exp_vit_32x16_l2_c3_wrule",
        "metric": {"goal": "maximize", "name": "Recall"},
        "parameters": {
            "patch_size": {"values": [4]},
            "hidden_dim": {"values": [8]},
            "num_heads": {"values": [8]},
            "num_layers": {"values": [2]},
            "learning_rate": {"values": [0.0001]},
            "row_th": {"values": [15]},
            "col_th": {"values": [7]},
            "channel": {"values": [3]},
            "num_epochs": {"values": [150]},
            "repeat_times": {"values": [i for i in range(10)] },
        }
    },
    "exp_vit_32x16_l4_c1_worule": {
        "method": "grid",
        "name": "exp_vit_32x16_l4_c1_worule",
        "metric": {"goal": "maximize", "name": "Recall"},
        "parameters": {
            "patch_size": {"values": [4]},
            "hidden_dim": {"values": [8]},
            "num_heads": {"values": [8]},
            "num_layers": {"values": [4]},
            "learning_rate": {"values": [0.0001]},
            "row_th": {"values": [15]},
            "col_th": {"values": [7]},
            "num_epochs": {"values": [150]},
            "channel": {"values": [1]},
            "repeat_times": {"values": [i for i in range(10)] },
        }
    },
    "exp_vit_32x16_l4_c1_wrule": {
        "method": "grid",
        "name": "exp_vit_32x16_l4_c1_wrule",
        "metric": {"goal": "maximize", "name": "Recall"},
        "parameters": {
            "patch_size": {"values": [4]},
            "hidden_dim": {"values": [8]},
            "num_heads": {"values": [8]},
            "num_layers": {"values": [4]},
            "learning_rate": {"values": [0.0001]},
            "row_th": {"values": [15]},
            "col_th": {"values": [7]},
            "channel": {"values": [1]},
            "num_epochs": {"values": [150]},
            "repeat_times": {"values": [i for i in range(10)] },
        }
    },
    "exp_vit_32x16_l4_c3_worule": {
        "method": "grid",
        "name": "exp_vit_32x16_l4_c3_worule",
        "metric": {"goal": "maximize", "name": "Recall"},
        "parameters": {
            "patch_size": {"values": [4]},
            "hidden_dim": {"values": [8]},
            "num_heads": {"values": [8]},
            "num_layers": {"values": [4]},
            "learning_rate": {"values": [0.0001]},
            "row_th": {"values": [15]},
            "col_th": {"values": [7]},
            "num_epochs": {"values": [150]},
            "channel": {"values": [3]},
            "repeat_times": {"values": [i for i in range(10)] },
        }
    },
    "exp_vit_32x16_l4_c3_wrule": {
        "method": "grid",
        "name": "exp_vit_32x16_l4_c3_wrule",
        "metric": {"goal": "maximize", "name": "Recall"},
        "parameters": {
            "patch_size": {"values": [4]},
            "hidden_dim": {"values": [8]},
            "num_heads": {"values": [8]},
            "num_layers": {"values": [4]},
            "learning_rate": {"values": [0.0001]},
            "row_th": {"values": [15]},
            "col_th": {"values": [7]},
            "channel": {"values": [3]},
            "num_epochs": {"values": [150]},
            "repeat_times": {"values": [i for i in range(10)] },
        }
    },
    # CNN 32 X 64
    "exp_cnn_64x32_l2_c1_worule": {
        "method": "grid",
        "name": "exp_cnn_64x32_l2_c1_worule",
        "metric": {"goal": "maximize", "name": "Recall"},
        "parameters": {
            "conv1_kernel_size": {"values": [15]},
            "conv2_kernel_size": {"values": [7]},
            "conv3_kernel_size": {"values": [5]},
            "conv4_kernel_size": {"values": [3]},
            "learning_rate": {"values": [0.0001]},
            "channel": {"values": [1]},
            "dropout": {"values": [0.2]},
            "conv1_out_channels": {"values": [64]},
            "conv2_out_channels": {"values": [128]},
            "row_th": {"values": [31]},
            "col_th": {"values": [15]},
            "num_epochs": {"values": [150]},
            "repeat_times": {"values": [i for i in range(5)] },
        }
    },
    "exp_cnn_64x32_l2_c1_wrule": {
        "method": "grid",
        "name": "exp_cnn_64x32_l2_c1_wrule",
        "metric": {"goal": "maximize", "name": "Recall"},
        "parameters": {
            "conv1_kernel_size": {"values": [15]},
            "conv2_kernel_size": {"values": [7]},
            "conv3_kernel_size": {"values": [5]},
            "conv4_kernel_size": {"values": [3]},
            "learning_rate": {"values": [0.0001]},
            "channel": {"values": [1]},
            "dropout": {"values": [0.2]},
            "conv1_out_channels": {"values": [64]},
            "conv2_out_channels": {"values": [128]},
            "row_th": {"values": [31]},
            "col_th": {"values": [15]},
            "num_epochs": {"values": [150]},
            "repeat_times": {"values": [i for i in range(5)] },
        }
    },
    "exp_cnn_64x32_l2_c3_worule": {
        "method": "grid",
        "name": "exp_cnn_64x32_l2_c3_worule",
        "metric": {"goal": "maximize", "name": "Recall"},
        "parameters": {
            "conv1_kernel_size": {"values": [15]},
            "conv2_kernel_size": {"values": [7]},
            "conv3_kernel_size": {"values": [5]},
            "conv4_kernel_size": {"values": [3]},
            "learning_rate": {"values": [0.0001]},
            "channel": {"values": [3]},
            "dropout": {"values": [0.2]},
            "conv1_out_channels": {"values": [64]},
            "conv2_out_channels": {"values": [128]},
            "row_th": {"values": [31]},
            "col_th": {"values": [15]},
            "num_epochs": {"values": [150]},
            "repeat_times": {"values": [i for i in range(5)] },
        }
    },
    "exp_cnn_64x32_l2_c3_wrule": {
        "method": "grid",
        "name": "exp_cnn_64x32_l2_c3_wrule",
        "metric": {"goal": "maximize", "name": "Recall"},
        "parameters": {
            "conv1_kernel_size": {"values": [15]},
            "conv2_kernel_size": {"values": [7]},
            "conv3_kernel_size": {"values": [5]},
            "conv4_kernel_size": {"values": [3]},
            "learning_rate": {"values": [0.0001]},
            "channel": {"values": [3]},
            "dropout": {"values": [0.2]},
            "conv1_out_channels": {"values": [64]},
            "conv2_out_channels": {"values": [128]},
            "row_th": {"values": [31]},
            "col_th": {"values": [15]},
            "num_epochs": {"values": [150]},
            "repeat_times": {"values": [i for i in range(5)] },
        }
    },
}

MODEL_NAME = "exp_resnet_32x16_l4_c1_wrule"

sweep_id = wandb.sweep(sweep_configs[MODEL_NAME], project="dram-hyperparameter-tuning")

class CustomDataset(Dataset):
    def __init__(self, data, labels):
        self.data = data
        self.labels = labels

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        sample = {"data": self.data[idx], "label": self.labels[idx]}
        return sample


class DRAMModel(object):
    def __init__(self):
        self.row_sections = 32
        self.col_sections = 16
        self.fixed_data = False
        # self._read_data()
        # self._down_sampling()
        # self._generate_data()
        # self._confirm_device()
        
    def _confirm_device(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def _read_data(self):
        logging.info("reading data started")
        self.feats = np.load(os.path.join(CONFIG.PATH_PROCESSED, f"feats_{self.row_sections}x{self.col_sections}_with_times.npy")).astype(np.float32)
        self.labels = np.load(os.path.join(CONFIG.PATH_PROCESSED, f"labels_{self.row_sections}x{self.col_sections}.npy"))
        # reflection to 0~1
        self.feats = 2 / (1 + np.exp(-self.feats)) - 1
        logging.info("reading data finished")
        
        logging.info("data sample")
        flattened_array = self.feats[555].reshape(-1, self.feats[555].shape[-1])
        np.save('numpy_array1.npy', flattened_array)

    def _read_data_multichannels(self):
        logging.info("reading data started")
        self.feats = np.load(os.path.join(CONFIG.PATH_PROCESSED, f"feats_{self.row_sections}x{self.col_sections}_with_times_multichannles.npy")).astype(np.float32)
        self.labels = np.load(os.path.join(CONFIG.PATH_PROCESSED, f"labels_{self.row_sections}x{self.col_sections}_with_times__multichannles.npy"))
        # reflection to 0~1
        new_feats = []
        new_labels = []
        for i in tqdm(range(len(self.feats))):
            feat = self.feats[i]
            random_number = int(random.choice(feat.flat))
            if random_number != 1:
                continue
            feat = (feat - random_number) * CONFIG.DRAM_MODEL_SCALING_PARAM
            feat = 2 / (1 + np.exp(-feat)) - 1
            new_feats.append(feat[0])

            label = self.labels[i]
            new_labels.append(label)
        self.feats = np.array(new_feats).astype(np.float32)
        self.labels = np.array(new_labels)
        logging.info(self.feats.shape)
        logging.info("reading data finished")
        
        

    # def _multichannel_to_singlechannel(self):
    #     logging.info("_multichannel_to_singlechannel started")
    #     new_feats = []
    #     new_labels = []
    #     for i in tqdm(range(len(self.feats))):
    #         feat = self.feats[i]
    #         random_number = int(random.choice(feat.flat))
    #         if random_number != 1:
    #             continue
    #         feat = (feat - random_number) * CONFIG.DRAM_MODEL_SCALING_PARAM
    #         feat = 2 / (1 + np.exp(-feat)) - 1
    #         new_feats.append(feat[0])

    #         label = self.labels[i]
    #         new_labels.append(label)
    #     self.feats = np.array(new_feats).astype(np.float32)
    #     self.labels = np.array(new_labels)
    #     logging.info("_multichannel_to_singlechannel finished")


    def _down_sampling(self):
        logging.info("down sampling started")
        # logging.info(f"{sum(self.labels)}")
        sampling_ratio = 0.1
        label_0_indices = np.where(self.labels == 0)[0]
        label_1_indices = np.where(self.labels == 1)[0]
        num_feats_to_select = int(sampling_ratio * len(self.feats))
        seed = int(random.random() * 100)
        logging.info(seed)
        np.random.seed(seed)
        selected_indices = np.random.choice(label_0_indices, size=num_feats_to_select, replace=False)
        logging.info(selected_indices)
        self.feats = np.concatenate((self.feats[selected_indices], self.feats[label_1_indices]))
        self.labels = np.concatenate((self.labels[selected_indices], self.labels[label_1_indices]))
        random_indices = np.arange(len(self.feats))
        np.random.shuffle(random_indices)
        self.feats = self.feats[random_indices]
        self.labels = self.labels[random_indices]
        logging.info("down sampling finished")

    def _generate_data(self):
        torch.manual_seed(0)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(0)
        np.random.seed(0)
        split_ratio = 0.8
        split = int(len(self.feats) * split_ratio)
        
        if not self.fixed_data:
            train_data, test_data = self.feats[:split], self.feats[split:]
            train_labels, test_labels = self.labels[:split], self.labels[split:]

            np.save(os.path.join(CONFIG.PATH_PROCESSED, f"train_feats_{self.row_sections}x{self.col_sections}.npy"), train_data)
            np.save(os.path.join(CONFIG.PATH_PROCESSED, f"train_labels_{self.row_sections}x{self.col_sections}.npy"), train_labels)
            np.save(os.path.join(CONFIG.PATH_PROCESSED, f"test_feats_{self.row_sections}x{self.col_sections}.npy"), test_data)
            np.save(os.path.join(CONFIG.PATH_PROCESSED, f"test_labels_{self.row_sections}x{self.col_sections}.npy"), test_labels)
            logging.info("loading new shuffle data")
        else:
            train_data = np.load(os.path.join(CONFIG.PATH_PROCESSED, f"train_feats_{self.row_sections}x{self.col_sections}.npy")).astype(np.float32)
            train_labels = np.load(os.path.join(CONFIG.PATH_PROCESSED, f"train_labels_{self.row_sections}x{self.col_sections}.npy"))
            test_data = np.load(os.path.join(CONFIG.PATH_PROCESSED, f"test_feats_{self.row_sections}x{self.col_sections}.npy")).astype(np.float32)
            test_labels = np.load(os.path.join(CONFIG.PATH_PROCESSED, f"test_labels_{self.row_sections}x{self.col_sections}.npy"))
            logging.info("loading existing data")        
        
        self.test_data = [item[0] for item in test_data.tolist()]
        batch_size = 64
        train_dataset = utils.CustomDataset(train_data, train_labels)
        test_dataset = utils.CustomDataset(test_data, test_labels)

        self.train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=False)
        self.test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    def _store_results(self):
        confusion = confusion_matrix(self.true_labels, self.predictions)
        tn, fp, fn, tp = confusion.ravel()

        tp_data = []
        fp_data = []
        tn_data = []
        fn_data = []

        for i in range(len(self.true_labels)):
            if self.true_labels[i] == 1 and self.predictions[i] == 1:
                tp_data.append(self.test_data[i])
            elif self.true_labels[i] == 0 and self.predictions[i] == 1:
                fp_data.append(self.test_data[i])
            elif self.true_labels[i] == 0 and self.predictions[i] == 0:
                tn_data.append(self.test_data[i])
            elif self.true_labels[i] == 1 and self.predictions[i] == 0:
                fn_data.append(self.test_data[i])

        with open(os.path.join(CONFIG.PATH_PROCESSED, f"_tp_data_{self.row_sections}x{self.col_sections}.csv"), "w+") as f:
            for item in tp_data:
                for row in item:
                    f.write(f"{row}\n")
                f.write("\n\n")

        with open(os.path.join(CONFIG.PATH_PROCESSED, f"_fp_data_{self.row_sections}x{self.col_sections}.csv"), "w+") as f:
            for item in fp_data:
                for row in item:
                    f.write(f"{row}\n")
                f.write("\n\n")

        with open(os.path.join(CONFIG.PATH_PROCESSED, f"_tn_data_{self.row_sections}x{self.col_sections}.csv"), "w+") as f:
            for item in tn_data:
                for row in item:
                    f.write(f"{row}\n")
                f.write("\n\n")

        with open(os.path.join(CONFIG.PATH_PROCESSED, f"_fn_data_{self.row_sections}x{self.col_sections}.csv"), "w+") as f:
            for item in fn_data:
                for row in item:
                    f.write(f"{row}\n")
                f.write("\n\n")

    def _apply_rules(self, data, row_th, col_th):
        non_zero_rows, non_zero_cols = np.nonzero(data)
        if len(non_zero_rows) > 0:
            max_non_zero_row_index = np.max(np.where(non_zero_rows))
            min_non_zero_row_index = np.min(np.where(non_zero_rows))
            row_count_between = max_non_zero_row_index - min_non_zero_row_index + 1
        else:
            row_count_between = 0
        if len(non_zero_cols) > 0:
            max_non_zero_col_index = np.max(non_zero_cols)
            min_non_zero_col_index = np.min(non_zero_cols)
            column_count_between = max_non_zero_row_index - min_non_zero_col_index + 1
        else: 
            column_count_between = 0

        if row_count_between >= row_th or column_count_between >= col_th:
            return 1
    
        return 0

    def train(self, model, criterion, optimizer, num_epochs=10):
        for epoch in range(num_epochs):
            model.train()
            running_loss = 0.0
            for batch in self.train_loader:
                inputs, labels = batch["data"].to(self.device), batch["label"].to(self.device)
                optimizer.zero_grad()
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()
                running_loss += loss.item()
                # wandb.log({"loss": running_loss})
            logging.info(f"Epoch {epoch + 1}/{num_epochs}, Loss: {running_loss / len(self.train_loader)}")
    
    def validate(self, model, row_th, col_th):
        model.eval()
        
        self.predictions = []
        self.true_labels = []
        """
        precision, recall on train set
        """
        correct = 0
        total = 0
        predictions = []
        true_labels = []
        with torch.no_grad():
            for batch in self.train_loader:
                inputs, labels = batch["data"].to(self.device), batch["label"].to(self.device)
                outputs = model(inputs)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
                predictions.extend(predicted.tolist())
                true_labels.extend(labels.tolist())
                logging.info(f"labels: {len(true_labels)}, positive: {true_labels.count(1)}")
        
        precision = precision_score(true_labels, predictions)
        recall = recall_score(true_labels, predictions)
        wandb.log({"train precision": precision, "train recall": recall})
        
        """
        precision, recall on test set
        """
        correct = 0
        total = 0
        predictions = []
        true_labels = []

        # positive_heatmaps = []
        # negative_heatmaps = []
        # target_layer = model.conv4
        # gradcam = GradCAM(model, target_layer)

        # model.to("cpu")
        # with torch.no_grad():
        for batch in self.test_loader:
            inputs, labels = batch["data"].to(self.device), batch["label"].to(self.device)
            # inputs, labels = batch["data"].to("cpu"), batch["label"].to("cpu")
            # inputs.requires_grad_(True)
            outputs = model(inputs)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

            # """
            # heatmap: add
            # """
            # for i in range(inputs.size(0)):
            #     if predicted[i] == labels[i]:
            #         # Correctly classified samples
            #         heatmap, _ = gradcam(inputs[i].unsqueeze(0))
            #         heatmap = heatmap.squeeze().numpy()  # Extract the heatmap and convert to NumPy array
            #         if predicted[i] == 1:
            #             positive_heatmaps.append(heatmap)
            #             positive_heatmap = heatmap
            #             # logging.info("pppppppppppppppppppppppppS")
            #             # logging.info(heatmap)
            #             # logging.info("pppppppppppppppppppppppppE")
            #         if predicted[i] == 0:
            #             # logging.info("eeeeeeeeeeeeeeeeeeeeeeeeeS")
            #             # logging.info(heatmap)
            #             # logging.info("eeeeeeeeeeeeeeeeeeeeeeeeeE")
            #             negative_heatmaps.append(heatmap)
            #             negative_heatmap = heatmap
            """
            rules
            """
            data = np.array(batch["data"])
            for i in range(len(data)):
                if labels[i] == 1 and predicted[i] == 0:
                    try:
                        pass
                        if "wrule" in MODEL_NAME:
                            predicted[i] = self._apply_rules(data[i][0], row_th, col_th)
                            logging.info("rules applied")
                        else:
                            logging.info("rules not applied")
                    except Exception as e:
                        # logging.error(f"{e}")
                        pass
            predictions.extend(predicted.tolist())
            true_labels.extend(labels.tolist())
        
        # """
        # heatmap: plot, save
        # """
        # average_positive_heatmap = np.mean(positive_heatmaps, axis=0)
        # average_negative_heatmap = np.mean(negative_heatmaps, axis=0)
        # # average_negative_heatmap = average_negative_heatmap

        # logging.info(average_positive_heatmap)

        # logging.info(average_negative_heatmap)

        # heatmap_difference = average_positive_heatmap - average_negative_heatmap
        # threshold = 0.01
        # heatmap_difference[heatmap_difference < threshold] = 0

        # plt.figure(figsize=(12, 6))
        # plt.subplot(1, 3, 1)
        # plt.imshow(positive_heatmap, cmap="jet")
        # plt.title("average heatmap of positive")
        # plt.axis("off")

        # plt.subplot(1, 3, 2)
        # plt.imshow(negative_heatmap, cmap="jet")
        # plt.title("average heatmap of negetive")
        # plt.axis("off")

        # plt.subplot(1, 3, 3)
        # plt.imshow(heatmap_difference, cmap="jet")
        # plt.title("heatmap difference")
        # plt.axis("off")

        # repeat_times = wandb.config.repeat_times
        # plt.savefig(f"heatmaps_conv4_{repeat_times}.png")


        self.predictions = predictions
        self.true_labels = true_labels
        self._store_results()

        precision = precision_score(true_labels, predictions)
        recall = recall_score(true_labels, predictions)
        f1 = f1_score(true_labels, predictions)

        with open(f"{MODEL_NAME}", "a") as f:
            f.writelines(f"Precision: {precision}, Recall: {recall}, F1: {f1}\n")

        logging.info(f"Precision: {precision}")
        logging.info(f"Recall: {recall}")
        logging.info(f"F1: {f1}")
        logging.info(f"Accuracy on test set: {(correct / total) * 100:.2f}%")
        return recall

    def validate_gradcam(self, model, row_th, col_th):
        model.eval()
        
        self.predictions = []
        self.true_labels = []
        
        """
        precision, recall on test set
        """
        correct = 0
        total = 0
        predictions = []
        true_labels = []

        positive_heatmaps = []
        negative_heatmaps = []
        target_layer = model.layer4
        gradcam = GradCAM(model, target_layer)
        
        positive_data = []
        negative_data = []

        # model.to("cpu")
        # with torch.no_grad():
        for batch in self.test_loader:
            inputs, labels = batch["data"].to(self.device), batch["label"].to(self.device)
            # inputs, labels = batch["data"].to("cpu"), batch["label"].to("cpu")
            # inputs.requires_grad_(True)
            outputs = model(inputs)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

            """
            heatmap: add
            """
            for i in range(inputs.size(0)):
                if predicted[i] == labels[i]:
                    # Correctly classified samples
                    heatmap, _ = gradcam(inputs[i].unsqueeze(0))
                    heatmap = heatmap.squeeze().numpy()  # Extract the heatmap and convert to NumPy array
                    if predicted[i] == 1:
                        positive_heatmaps.append(heatmap)
                        positive_heatmap = heatmap
                        # logging.info("pppppppppppppppppppppppppS")
                        # logging.info(heatmap)
                        # logging.info("pppppppppppppppppppppppppE")
                    if predicted[i] == 0:
                        # logging.info("eeeeeeeeeeeeeeeeeeeeeeeeeS")
                        # logging.info(heatmap)
                        # logging.info("eeeeeeeeeeeeeeeeeeeeeeeeeE")
                        negative_heatmaps.append(heatmap)
                        negative_heatmap = heatmap
                        
                        negative_data.append(inputs[i])
                        
            numpy_array = np.array(negative_data)
            np.savetxt('numpy_array.csv', numpy_array, delimiter=',')
            
            
            predictions.extend(predicted.tolist())
            true_labels.extend(labels.tolist())
        
        # """
        # heatmap: plot, save
        # """
        for i in tqdm(range(len(positive_heatmaps))):
            plt.figure(figsize=(8, 12))
            # plt.subplot(1, 3, 1)
            plt.imshow(positive_heatmaps[i], cmap="jet")
            # plt.title("average heatmap of positive")
            plt.axis("off")
            plt.colorbar()
            plt.savefig(f"../results/featuremaps/positive_heatmap_{i}.png")

        for i in tqdm(range(len(negative_heatmaps))):
            plt.figure(figsize=(8, 12))
            # plt.subplot(1, 3, 1)
            plt.imshow(negative_heatmaps[i], cmap="jet")
            # plt.title("average heatmap of positive")
            plt.axis("off")
            plt.colorbar()
            plt.savefig(f"../results/featuremaps/negative_heatmap_{i}.png")



        # average_positive_heatmap = np.mean(positive_heatmaps, axis=0)
        # average_negative_heatmap = np.mean(negative_heatmaps, axis=0)
        # average_positive_heatmap = positive_heatmaps[1]
        # average_negative_heatmap = negative_heatmaps[1]
        # # average_negative_heatmap = average_negative_heatmap

        # logging.info(average_positive_heatmap)

        # logging.info(average_negative_heatmap)

        # heatmap_difference = average_positive_heatmap - average_negative_heatmap
        # threshold = 0.01
        # heatmap_difference[heatmap_difference < threshold] = 0

        # plt.figure(figsize=(12, 6))
        # plt.subplot(1, 3, 1)
        # plt.imshow(positive_heatmap, cmap="jet")
        # plt.title("average heatmap of positive")
        # plt.axis("off")

        # plt.subplot(1, 3, 2)
        # plt.imshow(negative_heatmap, cmap="jet")
        # plt.title("average heatmap of negetive")
        # plt.axis("off")

        # plt.subplot(1, 3, 3)
        # plt.imshow(heatmap_difference, cmap="jet")
        # plt.title("heatmap difference")
        # plt.axis("off")

        # plt.savefig(f"../heatmaps_conv4_test.png")


        self.predictions = predictions
        self.true_labels = true_labels
        self._store_results()

        precision = precision_score(true_labels, predictions)
        recall = recall_score(true_labels, predictions)
        f1 = f1_score(true_labels, predictions)

        logging.info(f"Precision: {precision}")
        logging.info(f"Recall: {recall}")
        logging.info(f"F1: {f1}")
        logging.info(f"Accuracy on test set: {(correct / total) * 100:.2f}%")
        return recall
    
    def objective(self, trial):
        """
        for optuna
        """
        conv1_kernel_size = trial.suggest_int("conv1_kernel_size", 3, 17, step=2)
        conv2_kernel_size = trial.suggest_int("conv2_kernel_size", 3, 9, step=2)
        conv3_kernel_size = trial.suggest_int("conv3_kernel_size", 3, 7, step=2)
        conv4_kernel_size = trial.suggest_int("conv4_kernel_size", 3, 5, step=2)
        wandb.init(
            project="dram",
            name=f"model_cnn_{self.row_sections}x{self.col_sections}_{trial.number}",
            config={
                "conv1_kernel_size": conv1_kernel_size,
                "conv2_kernel_size": conv2_kernel_size,
                "conv3_kernel_size": conv3_kernel_size,
                "conv4_kernel_size": conv4_kernel_size
            }
        )

        model = md.CNNModel_64x32(conv1_kernel_size=conv1_kernel_size, conv2_kernel_size=conv2_kernel_size, \
                                  conv3_kernel_size=conv3_kernel_size, conv4_kernel_size=conv4_kernel_size)
        model.to(self.device)
        wandb.watch(model)
        
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters(), lr=0.001)

        self.train(model, criterion, optimizer, num_epochs=150)
        recall = self.validate(model)
        
        wandb.finish()

        return recall


    def sweep_cnn(self):
        
        self._read_data()
        self._down_sampling()
        self._generate_data()
        self._confirm_device()

        wandb.init()
        conv1_kernel_size = wandb.config.conv1_kernel_size
        conv2_kernel_size = wandb.config.conv2_kernel_size
        conv3_kernel_size = wandb.config.conv3_kernel_size
        conv4_kernel_size = wandb.config.conv4_kernel_size
        conv1_out_channels = wandb.config.conv1_out_channels
        conv2_out_channels = wandb.config.conv2_out_channels
        conv3_out_channels = wandb.config.conv3_out_channels
        conv4_out_channels = wandb.config.conv4_out_channels
        learning_rate = wandb.config.learning_rate
        dropout = wandb.config.dropout
        row_th = wandb.config.row_th
        col_th = wandb.config.col_th
        num_epochs = wandb.config.num_epochs

        model = md.CNNModel_32x16(conv1_kernel_size=conv1_kernel_size, conv2_kernel_size=conv2_kernel_size, \
                                  conv3_kernel_size=conv3_kernel_size, conv4_kernel_size=conv4_kernel_size, dropout=dropout, \
                                  conv1_out_channels=conv1_out_channels, conv2_out_channels=conv2_out_channels,\
                                  conv3_out_channels=conv3_out_channels, conv4_out_channels=conv4_out_channels)
        model.to(self.device)
        wandb.watch(model)
        
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters(), lr=learning_rate)

        self.train(model, criterion, optimizer, num_epochs=num_epochs)
        recall = self.validate(model, row_th, col_th)
        wandb.log({"Recall": recall})
        wandb.log({"Metric": row_th * 100 + col_th})

    def sweep_vit(self):
        
        wandb.init()

        learning_rate = wandb.config.learning_rate
        patch_size = wandb.config.patch_size
        hidden_dim = wandb.config.hidden_dim
        num_heads = wandb.config.num_heads
        num_layers = wandb.config.num_layers

        image_size = (1, self.row_sections, self.col_sections)
        num_classes = 2

        model = md.ViT(image_size, num_classes, patch_size=(patch_size,patch_size), hidden_dim=hidden_dim, num_heads=num_heads, num_layers=num_layers)

        model.to(self.device)
        wandb.watch(model)
        
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters(), lr=learning_rate)

        self.train(model, criterion, optimizer, num_epochs=200)
        recall = self.validate(model)
        wandb.log({"Recall": recall})
        

    def sweep_resnet(self):
        self._read_data()
        self._down_sampling()
        self._generate_data()
        self._confirm_device()
        wandb.init()
        conv1_kernel_size = wandb.config.conv1_kernel_size
        conv2_kernel_size = wandb.config.conv2_kernel_size
        conv3_kernel_size = wandb.config.conv3_kernel_size
        conv4_kernel_size = wandb.config.conv4_kernel_size
        conv1_out_channels = wandb.config.conv1_out_channels
        conv2_out_channels = wandb.config.conv2_out_channels
        conv3_out_channels = wandb.config.conv3_out_channels
        conv4_out_channels = wandb.config.conv4_out_channels
        learning_rate = wandb.config.learning_rate
        dropout = wandb.config.dropout
        row_th = wandb.config.row_th
        col_th = wandb.config.col_th
        num_epochs = wandb.config.num_epochs

        model = md.ResNet4_32x16(conv1_kernel_size=conv1_kernel_size, conv2_kernel_size=conv2_kernel_size, \
                                  conv3_kernel_size=conv3_kernel_size, conv4_kernel_size=conv4_kernel_size)
        model.to(self.device)
        model = nn.DataParallel(model)
        wandb.watch(model)
        
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters(), lr=learning_rate)

        self.train(model, criterion, optimizer, num_epochs=num_epochs)
        recall = self.validate(model, row_th, col_th)
        wandb.log({"Recall": recall})


    def sweep_resnet_multichannels(self):
        self._read_data_multichannels()
        self._down_sampling()
        self._generate_data()
        self._confirm_device()
        wandb.init()
        conv1_kernel_size = wandb.config.conv1_kernel_size
        conv2_kernel_size = wandb.config.conv2_kernel_size
        conv3_kernel_size = wandb.config.conv3_kernel_size
        conv4_kernel_size = wandb.config.conv4_kernel_size
        conv1_out_channels = wandb.config.conv1_out_channels
        conv2_out_channels = wandb.config.conv2_out_channels
        conv3_out_channels = wandb.config.conv3_out_channels
        conv4_out_channels = wandb.config.conv4_out_channels
        learning_rate = wandb.config.learning_rate
        dropout = wandb.config.dropout
        row_th = wandb.config.row_th
        col_th = wandb.config.col_th
        num_epochs = wandb.config.num_epochs

        model = md.ResNet4_32x16_multichannels(conv1_kernel_size=conv1_kernel_size, conv2_kernel_size=conv2_kernel_size, \
                                  conv3_kernel_size=conv3_kernel_size, conv4_kernel_size=conv4_kernel_size)
        model.to(self.device)
        model = nn.DataParallel(model)
        wandb.watch(model)
        
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters(), lr=learning_rate)

        self.train(model, criterion, optimizer, num_epochs=num_epochs)
        recall = self.validate(model, row_th, col_th)
        wandb.log({"Recall": recall})
    

    def plain_training(self):
        self._read_data()
        self._down_sampling()
        self._generate_data()
        self._confirm_device()
        model = md.ResNet4_32x16(conv1_kernel_size=9, conv2_kernel_size=5, \
                                  conv3_kernel_size=3, conv4_kernel_size=3)
        model.to(self.device)
        # model = nn.DataParallel(model)
        
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters(), lr=0.0002)

        self.train(model, criterion, optimizer, num_epochs=150)
        torch.save(model.state_dict(), os.path.join(CONFIG.PATH_MODEL, f"{MODEL_NAME}-{CONFIG.MODEL_VERSION_TEST}"))
        # recall = self.validate(model, 15, 5)


    def plain_training_resnet_multichannels(self):
        self._read_data_multichannels()
        self._down_sampling()
        self._generate_data()
        self._confirm_device()
        model = md.ResNet4_32x16_multichannels(conv1_kernel_size=9, conv2_kernel_size=5, \
                                  conv3_kernel_size=3, conv4_kernel_size=3)
        model.to(self.device)
        # model = nn.DataParallel(model)
        
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters(), lr=0.0002)

        self.train(model, criterion, optimizer, num_epochs=150)
        torch.save(model.state_dict(), os.path.join(CONFIG.PATH_MODEL, f"{MODEL_NAME}-{CONFIG.MODEL_VERSION_TEST}"))


    """
    functions for results
    """
    def exp_cnn_32x16_l2_c1_wowrule(self):
        
        self._read_data()
        self._down_sampling()
        self._generate_data()
        self._confirm_device()

        wandb.init()
        conv1_kernel_size = wandb.config.conv1_kernel_size
        conv2_kernel_size = wandb.config.conv2_kernel_size
        conv3_kernel_size = wandb.config.conv3_kernel_size
        conv4_kernel_size = wandb.config.conv4_kernel_size
        conv1_out_channels = wandb.config.conv1_out_channels
        conv2_out_channels = wandb.config.conv2_out_channels
        learning_rate = wandb.config.learning_rate
        dropout = wandb.config.dropout
        row_th = wandb.config.row_th
        col_th = wandb.config.col_th
        num_epochs = wandb.config.num_epochs
        channel = wandb.config.channel

        model = md.CNNModel_32x16_two(channel=channel, conv1_kernel_size=conv1_kernel_size, conv2_kernel_size=conv2_kernel_size, \
                                  dropout=dropout, \
                                  conv1_out_channels=conv1_out_channels, conv2_out_channels=conv2_out_channels)
        model.to(self.device)
        wandb.watch(model)
        
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters(), lr=learning_rate)

        self.train(model, criterion, optimizer, num_epochs=num_epochs)
        recall = self.validate(model, row_th, col_th)
        wandb.log({"Recall": recall})
        wandb.log({"Metric": row_th * 100 + col_th})
    
    def exp_cnn_32x16_l2_c3_wowrule(self):
        
        self._read_data_multichannels()
        self._down_sampling()
        self._generate_data()
        self._confirm_device()

        wandb.init()
        conv1_kernel_size = wandb.config.conv1_kernel_size
        conv2_kernel_size = wandb.config.conv2_kernel_size
        conv3_kernel_size = wandb.config.conv3_kernel_size
        conv4_kernel_size = wandb.config.conv4_kernel_size
        conv1_out_channels = wandb.config.conv1_out_channels
        conv2_out_channels = wandb.config.conv2_out_channels
        learning_rate = wandb.config.learning_rate
        dropout = wandb.config.dropout
        row_th = wandb.config.row_th
        col_th = wandb.config.col_th
        num_epochs = wandb.config.num_epochs
        channel = wandb.config.channel

        model = md.CNNModel_32x16_two(channel=channel, conv1_kernel_size=conv1_kernel_size, conv2_kernel_size=conv2_kernel_size, \
                                  dropout=dropout, \
                                  conv1_out_channels=conv1_out_channels, conv2_out_channels=conv2_out_channels)
        model.to(self.device)
        wandb.watch(model)
        
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters(), lr=learning_rate)

        self.train(model, criterion, optimizer, num_epochs=num_epochs)
        recall = self.validate(model, row_th, col_th)
        wandb.log({"Recall": recall})
        wandb.log({"Metric": row_th * 100 + col_th})

    def exp_cnn_32x16_l4_c1_wowrule(self):
        self._read_data()
        self._down_sampling()
        self._generate_data()
        self._confirm_device()

        wandb.init()
        conv1_kernel_size = wandb.config.conv1_kernel_size
        conv2_kernel_size = wandb.config.conv2_kernel_size
        conv3_kernel_size = wandb.config.conv3_kernel_size
        conv4_kernel_size = wandb.config.conv4_kernel_size
        conv1_out_channels = wandb.config.conv1_out_channels
        conv2_out_channels = wandb.config.conv2_out_channels
        conv3_out_channels = wandb.config.conv3_out_channels
        conv4_out_channels = wandb.config.conv4_out_channels
        learning_rate = wandb.config.learning_rate
        dropout = wandb.config.dropout
        row_th = wandb.config.row_th
        col_th = wandb.config.col_th
        num_epochs = wandb.config.num_epochs
        channel = wandb.config.channel

        model = md.CNNModel_32x16(channel=channel, conv1_kernel_size=conv1_kernel_size, conv2_kernel_size=conv2_kernel_size, \
                                    conv3_kernel_size=conv3_kernel_size, conv4_kernel_size=conv4_kernel_size, \
                                  dropout=dropout, \
                                  conv1_out_channels=conv1_out_channels, conv2_out_channels=conv2_out_channels,\
                                  conv3_out_channels=conv3_out_channels, conv4_out_channels=conv4_out_channels)
        model.to(self.device)
        wandb.watch(model)
        
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters(), lr=learning_rate)

        self.train(model, criterion, optimizer, num_epochs=num_epochs)
        recall = self.validate(model, row_th, col_th)
        wandb.log({"Recall": recall})
        wandb.log({"Metric": row_th * 100 + col_th})

    def exp_cnn_32x16_l4_c3_worule(self):
        self._read_data_multichannels()
        self._down_sampling()
        self._generate_data()
        self._confirm_device()

        wandb.init()
        conv1_kernel_size = wandb.config.conv1_kernel_size
        conv2_kernel_size = wandb.config.conv2_kernel_size
        conv3_kernel_size = wandb.config.conv3_kernel_size
        conv4_kernel_size = wandb.config.conv4_kernel_size
        conv1_out_channels = wandb.config.conv1_out_channels
        conv2_out_channels = wandb.config.conv2_out_channels
        conv3_out_channels = wandb.config.conv3_out_channels
        conv4_out_channels = wandb.config.conv4_out_channels
        learning_rate = wandb.config.learning_rate
        dropout = wandb.config.dropout
        row_th = wandb.config.row_th
        col_th = wandb.config.col_th
        num_epochs = wandb.config.num_epochs
        channel = wandb.config.channel

        model = md.CNNModel_32x16(channel=channel, conv1_kernel_size=conv1_kernel_size, conv2_kernel_size=conv2_kernel_size, \
                                    conv3_kernel_size=conv3_kernel_size, conv4_kernel_size=conv4_kernel_size, \
                                  dropout=dropout, \
                                  conv1_out_channels=conv1_out_channels, conv2_out_channels=conv2_out_channels,\
                                  conv3_out_channels=conv3_out_channels, conv4_out_channels=conv4_out_channels)
        model.to(self.device)
        wandb.watch(model)
        
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters(), lr=learning_rate)

        self.train(model, criterion, optimizer, num_epochs=num_epochs)
        recall = self.validate(model, row_th, col_th)
        wandb.log({"Recall": recall})
        wandb.log({"Metric": row_th * 100 + col_th})

    def exp_resnet_32x16_l2_c1_wowrule(self):
        self._read_data()
        self._down_sampling()
        self._generate_data()
        self._confirm_device()

        wandb.init()
        conv1_kernel_size = wandb.config.conv1_kernel_size
        conv2_kernel_size = wandb.config.conv2_kernel_size
        conv3_kernel_size = wandb.config.conv3_kernel_size
        conv4_kernel_size = wandb.config.conv4_kernel_size
        conv1_out_channels = wandb.config.conv1_out_channels
        conv2_out_channels = wandb.config.conv2_out_channels
        conv3_out_channels = wandb.config.conv3_out_channels
        conv4_out_channels = wandb.config.conv4_out_channels
        learning_rate = wandb.config.learning_rate
        dropout = wandb.config.dropout
        row_th = wandb.config.row_th
        col_th = wandb.config.col_th
        num_epochs = wandb.config.num_epochs
        channel = wandb.config.channel

        with open(f"{MODEL_NAME}", "a") as f:
            f.writelines(f"row_th: {row_th}, col_th: {col_th}, ")

        model = md.ResNet4_32x16_Two(channel=channel, conv1_kernel_size=conv1_kernel_size, conv2_kernel_size=conv2_kernel_size, \
                                    conv3_kernel_size=conv3_kernel_size, conv4_kernel_size=conv4_kernel_size, \
                                  dropout=dropout)
        model.to(self.device)
        wandb.watch(model)
        
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters(), lr=learning_rate)

        self.train(model, criterion, optimizer, num_epochs=num_epochs)
        recall = self.validate(model, row_th, col_th)
        wandb.log({"Recall": recall})
        wandb.log({"Metric": row_th * 100 + col_th})


    def exp_resnet_32x16_l2_c3_wowrule(self):
        self._read_data_multichannels()
        self._down_sampling()
        self._generate_data()
        self._confirm_device()

        wandb.init()
        conv1_kernel_size = wandb.config.conv1_kernel_size
        conv2_kernel_size = wandb.config.conv2_kernel_size
        conv3_kernel_size = wandb.config.conv3_kernel_size
        conv4_kernel_size = wandb.config.conv4_kernel_size
        conv1_out_channels = wandb.config.conv1_out_channels
        conv2_out_channels = wandb.config.conv2_out_channels
        conv3_out_channels = wandb.config.conv3_out_channels
        conv4_out_channels = wandb.config.conv4_out_channels
        learning_rate = wandb.config.learning_rate
        dropout = wandb.config.dropout
        row_th = wandb.config.row_th
        col_th = wandb.config.col_th
        num_epochs = wandb.config.num_epochs
        repeat_times = wandb.config.repeat_times
        channel = wandb.config.channel

        with open(f"{MODEL_NAME}", "a") as f:
            f.writelines(f"row_th: {row_th}, col_th: {col_th}, ")

        model = md.ResNet4_32x16_Two(channel=channel, conv1_kernel_size=conv1_kernel_size, conv2_kernel_size=conv2_kernel_size, \
                                    conv3_kernel_size=conv3_kernel_size, conv4_kernel_size=conv4_kernel_size, \
                                  dropout=dropout)
        model.to(self.device)
        wandb.watch(model)
        
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters(), lr=learning_rate)

        self.train(model, criterion, optimizer, num_epochs=num_epochs)
        recall = self.validate(model, row_th, col_th)
        wandb.log({"Recall": recall})
        wandb.log({"Metric": row_th * 100 + col_th})

    def exp_resnet_32x16_l4_c1_wowrule(self):
        self._read_data()
        self._down_sampling()
        self._generate_data()
        self._confirm_device()

        wandb.init()
        conv1_kernel_size = wandb.config.conv1_kernel_size
        conv2_kernel_size = wandb.config.conv2_kernel_size
        conv3_kernel_size = wandb.config.conv3_kernel_size
        conv4_kernel_size = wandb.config.conv4_kernel_size
        conv1_out_channels = wandb.config.conv1_out_channels
        conv2_out_channels = wandb.config.conv2_out_channels
        conv3_out_channels = wandb.config.conv3_out_channels
        conv4_out_channels = wandb.config.conv4_out_channels
        learning_rate = wandb.config.learning_rate
        dropout = wandb.config.dropout
        row_th = wandb.config.row_th
        col_th = wandb.config.col_th
        num_epochs = wandb.config.num_epochs
        channel = wandb.config.channel

        model = md.ResNet4_32x16(channel=channel, conv1_kernel_size=conv1_kernel_size, conv2_kernel_size=conv2_kernel_size, \
                                    conv3_kernel_size=conv3_kernel_size, conv4_kernel_size=conv4_kernel_size, \
                                  dropout=dropout)
        model.to(self.device)
        wandb.watch(model)
        
        torch.save(model.state_dict(), "resnet_l4_c1.pth")
        
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters(), lr=learning_rate)

        self.train(model, criterion, optimizer, num_epochs=num_epochs)
        recall = self.validate(model, row_th, col_th)
        wandb.log({"Recall": recall})
        wandb.log({"Metric": row_th * 100 + col_th})

    def exp_resnet_32x16_l4_c3_wowrule(self):
        self._read_data_multichannels()
        self._down_sampling()
        self._generate_data()
        self._confirm_device()

        wandb.init()
        conv1_kernel_size = wandb.config.conv1_kernel_size
        conv2_kernel_size = wandb.config.conv2_kernel_size
        conv3_kernel_size = wandb.config.conv3_kernel_size
        conv4_kernel_size = wandb.config.conv4_kernel_size
        conv1_out_channels = wandb.config.conv1_out_channels
        conv2_out_channels = wandb.config.conv2_out_channels
        conv3_out_channels = wandb.config.conv3_out_channels
        conv4_out_channels = wandb.config.conv4_out_channels
        learning_rate = wandb.config.learning_rate
        dropout = wandb.config.dropout
        row_th = wandb.config.row_th
        col_th = wandb.config.col_th
        num_epochs = wandb.config.num_epochs
        repeat_times = wandb.config.repeat_times
        channel = wandb.config.channel

        with open(f"{MODEL_NAME}", "a") as f:
            f.writelines(f"row_th: {row_th}, col_th: {col_th}, ")

        model = md.ResNet4_32x16(channel=channel, conv1_kernel_size=conv1_kernel_size, conv2_kernel_size=conv2_kernel_size, \
                                    conv3_kernel_size=conv3_kernel_size, conv4_kernel_size=conv4_kernel_size, \
                                  dropout=dropout)
        model.to(self.device)
        wandb.watch(model)
        
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters(), lr=learning_rate)

        self.train(model, criterion, optimizer, num_epochs=num_epochs)
        recall = self.validate(model, row_th, col_th)
        wandb.log({"Recall": recall})
        wandb.log({"Metric": row_th * 100 + col_th})

    def exp_vit_32x16_l2_c1_wowrule(self):
        self._read_data()
        self._down_sampling()
        self._generate_data()
        self._confirm_device()
        wandb.init()

        learning_rate = wandb.config.learning_rate
        patch_size = wandb.config.patch_size
        hidden_dim = wandb.config.hidden_dim
        num_heads = wandb.config.num_heads
        num_layers = wandb.config.num_layers
        row_th = wandb.config.row_th
        col_th = wandb.config.col_th
        repeat_times = wandb.config.repeat_times
        num_epochs = wandb.config.num_epochs

        image_size = (1, self.row_sections, self.col_sections)
        num_classes = 2

        model = md.ViT(image_size, num_classes, patch_size=(patch_size,patch_size), hidden_dim=hidden_dim, num_heads=num_heads, num_layers=num_layers)

        model.to(self.device)
        wandb.watch(model)
        
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters(), lr=learning_rate)

        self.train(model, criterion, optimizer, num_epochs=num_epochs)
        recall = self.validate(model, row_th, col_th)
        wandb.log({"Recall": recall})

    def exp_vit_wowrule(self):
        # self._read_data_multichannels()
        wandb.init()

        learning_rate = wandb.config.learning_rate
        patch_size = wandb.config.patch_size
        hidden_dim = wandb.config.hidden_dim
        num_heads = wandb.config.num_heads
        num_layers = wandb.config.num_layers
        row_th = wandb.config.row_th
        col_th = wandb.config.col_th
        repeat_times = wandb.config.repeat_times
        num_epochs = wandb.config.num_epochs
        channel = wandb.config.channel

        if channel == 1:
            self._read_data()
        else:
            self._read_data_multichannels()
        self._down_sampling()
        self._generate_data()
        self._confirm_device()

        image_size = (channel, self.row_sections, self.col_sections)
        num_classes = 2

        model = md.ViT(image_size, num_classes, patch_size=(patch_size,patch_size), hidden_dim=hidden_dim, num_heads=num_heads, num_layers=num_layers)

        model.to(self.device)
        wandb.watch(model)
        
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters(), lr=learning_rate)

        self.train(model, criterion, optimizer, num_epochs=num_epochs)
        recall = self.validate(model, row_th, col_th)
        wandb.log({"Recall": recall})

    """
    test
    """
    def plain_training_test(self):
        self._read_data_multichannels()
        self._down_sampling()
        self._generate_data()
        self._confirm_device()

        model = md.CNNModel_32x16_two(channel=channel, conv1_kernel_size=conv1_kernel_size, conv2_kernel_size=conv2_kernel_size, \
                                    conv3_kernel_size=conv3_kernel_size, conv4_kernel_size=conv4_kernel_size, \
                                  dropout=dropout, \
                                  conv1_out_channels=conv1_out_channels, conv2_out_channels=conv2_out_channels,\
                                  conv3_out_channels=conv3_out_channels, conv4_out_channels=conv4_out_channels)
        model.to(self.device)
        wandb.watch(model)
        
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters(), lr=learning_rate)

        self.train(model, criterion, optimizer, num_epochs=num_epochs)
        recall = self.validate(model, row_th, col_th)
        wandb.log({"Recall": recall})
        wandb.log({"Metric": row_th * 100 + col_th})

    """
    feature map
    """
    def feature_map(self):
        self._read_data()
        # self._read_data_multichannels()
        self._down_sampling()
        self._generate_data()
        self._confirm_device()
        self.device = torch.device("cpu")

        model = md.ResNet4_32x16(conv1_kernel_size=9, conv2_kernel_size=5, \
                                  conv3_kernel_size=3, conv4_kernel_size=3)
        model.load_state_dict(torch.load(os.path.join(CONFIG.PATH_MODEL, "resnet-ConvRule-model-0.1.pth")))
        model.to(self.device)
        model.eval()

        self.validate_gradcam(model, 2, 3)


if __name__ == "__main__":
    t1 = time.time()

    """
    plain, to generate SOAT model using optimal parameters
    """
    if False:
        dram_model = DRAMModel()
        dram_model.plain_training_test()
        sys.exit()

    """
    heatmap, to generate heatmap
    """
    if True:
        logging.info("generating feature maps...")
        dram_model = DRAMModel()
        dram_model.feature_map()
        sys.exit()
    
    """
    optuna
    """
    # dram_model = DRAMModel()
    # study = optuna.create_study(study_name="cnn_64x32", storage="sqlite:///db.sqlite3", direction="maximize")
    # study.optimize(dram_model.objective, n_trials=100)
    # best_params = study.best_params
    # best_recall = study.best_value
    # print(f"Best Parameters: {best_params}")
    # print(f"Best Recall: {best_recall}")
    
    # """
    # wandb
    # """
    dram_model = DRAMModel()
    wandb.agent(sweep_id, function=dram_model.exp_resnet_32x16_l4_c1_wowrule)
    
    t2 = time.time()
    logging.info(f"running time: {(t2 - t1) / 60} mins")