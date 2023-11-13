import torch
import os

MOUNT_ROOT_DIR = "/exp"

CONFIG_FILE = os.path.join(MOUNT_ROOT_DIR, "Config/config.json")
LOG_FILE = os.path.join(MOUNT_ROOT_DIR, "Logs/detr.log")
DATASET_LOCATION = os.path.join(MOUNT_ROOT_DIR, "Dataset")
MODEL_PATH = os.path.join(MOUNT_ROOT_DIR, "Weights/Trained/")
PREVIOUS_TRAIN_PATH = os.path.join(MOUNT_ROOT_DIR, "Weights/Pretrained/")
ANNOTATION_FILE_NAME = "_annotations.coco.json"
TRAIN_DIRECTORY = os.path.join(DATASET_LOCATION, "train")
VAL_DIRECTORY = os.path.join(DATASET_LOCATION, "valid")
TEST_DIRECTORY = os.path.join(DATASET_LOCATION, "test")
PREDICTION_DIR = os.path.join(MOUNT_ROOT_DIR, "Predictions")


DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
CHECKPOINT = "facebook/detr-resnet-50"
