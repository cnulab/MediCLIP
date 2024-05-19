
import logging
from datetime import datetime
import random
import os
import torch
import numpy as np
from collections.abc import Mapping
import shutil

from sklearn import metrics


def map_func(storage, location):
    return storage.cuda()


def create_logger(name, log_file, level=logging.INFO):
    log = logging.getLogger(name)
    formatter = logging.Formatter(
        "[%(asctime)s][%(filename)15s][line:%(lineno)4d][%(levelname)8s] %(message)s"
    )
    fh = logging.FileHandler(log_file)
    fh.setFormatter(formatter)
    sh = logging.StreamHandler()
    sh.setFormatter(formatter)
    log.setLevel(level)
    log.addHandler(fh)
    log.addHandler(sh)
    return log



def set_seed(seed):
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    random.seed(seed)



def get_current_time():
    current_time = datetime.now().strftime("%Y_%m_%d_%H_%M_%S")
    return current_time



class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self, length=0):
        self.length = length
        self.reset()

    def reset(self):
        if self.length > 0:
            self.history = []
        else:
            self.count = 0
            self.sum = 0.0
        self.val = 0.0
        self.avg = 0.0

    def update(self, val, num=1):
        if self.length > 0:
            # currently assert num==1 to avoid bad usage, refine when there are some explict requirements
            assert num == 1
            self.history.append(val)
            if len(self.history) > self.length:
                del self.history[0]

            self.val = self.history[-1]
            self.avg = np.mean(self.history)
        else:
            self.val = val
            self.sum += val * num
            self.count += num
            self.avg = self.sum / self.count


def compute_imagewise_metrics(
    anomaly_prediction,
    anomaly_ground_truth_labels
):
    """
    Computes retrieval statistics (AUROC, FPR, TPR).

    Args:
        anomaly_prediction: [np.array or list] [N] Assignment weights
                                    per image. Higher indicates higher
                                    probability of being an anomaly.
        anomaly_ground_truth_labels: [np.array or list] [N] Binary labels - 1
                                    if image is an anomaly, 0 if not.
    """
    auroc = metrics.roc_auc_score(
        anomaly_ground_truth_labels, anomaly_prediction
    )

    return {"image-auroc": auroc}


def compute_pixelwise_metrics(
    pixel_prediction,
    pixel_ground_truth_labels
):
    """
    Computes retrieval statistics (AUROC, FPR, TPR).

    Args:
        anomaly_prediction: [np.array or list] [N] Assignment weights
                                    per image. Higher indicates higher
                                    probability of being an anomaly.
        anomaly_ground_truth_labels: [np.array or list] [N] Binary labels - 1
                                    if image is an anomaly, 0 if not.
    """
    pixel_prediction = np.concatenate(
        [pred.flatten() for pred in pixel_prediction], axis=0
    )

    pixel_ground_truth_labels = np.concatenate(
        [label.flatten() for label in pixel_ground_truth_labels], axis=0
    )

    pixel_ground_truth_labels[pixel_ground_truth_labels > 0] = 1

    pixel_auroc = metrics.roc_auc_score(
        pixel_ground_truth_labels, pixel_prediction
    )

    return {"pixel-auroc": pixel_auroc}