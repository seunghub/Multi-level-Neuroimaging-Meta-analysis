import joblib
import numpy as np
import pandas as pd
import nibabel as nib
import torch
from functools import partial
from joblib import delayed
from loguru import logger
from pathlib import Path

from neuroquery.img_utils import get_masker, iter_coordinates_to_maps
from neuroquery import datasets as nq_datasets
from nilearn import datasets, image
from nilearn.input_data import NiftiMasker
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from src.constants import DATA_PATH
from src.parallel import ParallelExecutor
from src.utils import clip_predict, recall_n

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def recall_n_callback(loader, num_samples=None, device=DEVICE):
    def run_callback(model, epoch_index, n=10):
        model.eval()

        with torch.no_grad():
            text_probs = clip_predict(
                model,
                loader,
                num_samples=num_samples,
                device=device,
            )

        recall = recall_n(
            text_probs,
            np.eye(len(text_probs)),
            n_first=n,
            thresh=0.95,
            reduce_mean=True,
        )
        return recall

    return run_callback


def diagonal_callback(loader, device=DEVICE):
    # mean of the diagonal
    # diagonal dominance? ratio of diagonal over sum(non-diagonal terms)
    def run_callback(model, epoch_index):
        model.eval()

        with torch.no_grad():
            text_probs = clip_predict(
                model,
                loader,
                num_samples=None,
                device=device,
            )

        return np.mean(np.diag(text_probs))

    return run_callback

def non_diagonal_callback(loader, device=DEVICE):
    def run_callback(model, epoch_index):
        model.eval()

        with torch.no_grad():
            text_probs = clip_predict(
                model,
                loader,
                num_samples=None,
                device=device,
            )

        return np.mean(text_probs - np.diag(np.diag(text_probs)))

    return run_callback

def term_to_one_callback(loader, device=DEVICE):
    def run_callback(model, epoch_index):
        model.eval()

        with torch.no_grad():
            text_probs = clip_predict(
                model,
                loader,
                num_samples=None,
                device=device,
            )

        text_probs[text_probs > 0.9999] = 1
        text_probs[text_probs < 0.9999] = 0

        return np.mean(text_probs.sum(axis=1) / len(text_probs))

    return run_callback


