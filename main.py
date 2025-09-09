import os
import sys
import pandas as pd
import pickle
from collections import defaultdict
from functools import partial
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from sklearn.decomposition import PCA
from sklearn.model_selection import KFold
from sklearn.pipeline import Pipeline
from layers import ClipModel, MLP, ProjectionHead, ResidualHead, MeruModel, OurModel
from losses import ClipLoss, MeruLoss
from plotting import plot_matrix
from training import (
    check_model_parameter_callback, count_parameters,
    diagonal_callback, non_diagonal_callback,
    predict, recall_n_callback, train,
)
from sklearn.preprocessing import Normalizer, StandardScaler
from torch.utils.data import DataLoader, TensorDataset
from metrics import mix_match
from src.utils import plot_training, recall_n
import argparse
import lorentz as L
import random
from scipy.stats import kendalltau
from mpl_toolkits.mplot3d.art3d import Line3D
from mpl_toolkits.mplot3d import Axes3D

parser = argparse.ArgumentParser(description="NeuroContext Test.")
parser.add_argument("--model_type", type=str, required=True, 
                    help="CLIP / MERU / OURS")
parser.add_argument("--device", type=int, default=0)
parser.add_argument("--save_idx", type=int, default=0)
args = parser.parse_args()

# def set_seed(seed: int = 42, cudnn_deterministic: bool = True) -> None:
def set_seed(seed: int = 42, cudnn_deterministic: bool = True) -> None:
    os.environ['PYTHONHASHSEED'] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    if cudnn_deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

if __name__ == '__main__':
    set_seed()

    # Get the current working directory
    current_folder_path = os.getcwd()

    # Get the parent directory
    parent_folder_path = os.path.dirname(current_folder_path)

    # Append both directories to sys.path
    sys.path.append(current_folder_path)
    sys.path.append(parent_folder_path)

    # Change the current working directory to the current directory (optional, since it's already the current directory)
    os.chdir(current_folder_path)

    print("Current Working Directory: ", os.getcwd())
    print("sys.path: ", sys.path)


    # Define the directory where the data files are saved
    current_directory = os.getcwd()
    data_dir = os.path.join(current_directory, 'data')

    # List all files in the directory
    files = os.listdir(data_dir)

    # Dictionary to store the loaded data
    loaded_data = {}

    # Load each file and store it in the dictionary
    for file in files:
        if file.endswith('.pkl'):
            var_name = file.replace('.pkl', '')
            print(f"Loading {var_name}")
            with open(os.path.join(data_dir, file), 'rb') as f:
                loaded_data[var_name] = pickle.load(f)

    # Unpack the loaded data into variables
    globals().update(loaded_data)

    # Verify required data is loaded
    required_vars = ['preprocessed_train_text_embeddings', 'preprocessed_train_gaussian_embeddings',
                    'preprocessed_test_text_embeddings', 'preprocessed_test_gaussian_embeddings']
    missing_vars = [var for var in required_vars if var not in globals()]
    if missing_vars:
        raise ValueError(f"Missing required data: {missing_vars}")

    print("All data files have been loaded.")
    # import required modules

    # training the model
    plot_verbose = True
    batch_size = 4096
    lr = 1e-4 #1e-4
    weight_decay = 0.05 #0.1
    dropout = 0.6
    num_epochs = 200
    entail_weight = 0.5
    output_size = preprocessed_test_gaussian_embeddings.shape[1]

    # with open('data/train_title_embeddings.pkl','rb') as f: preprocessed_train_text_embeddings = pickle.load(f)
    # with open('data/test_title_embeddings.pkl','rb') as f: preprocessed_test_text_embeddings = pickle.load(f)
    # with open('data/train_abstract_embeddings.pkl','rb') as f: preprocessed_train_text_embeddings = pickle.load(f)
    # with open('data/test_abstract_embeddings.pkl','rb') as f: preprocessed_test_text_embeddings = pickle.load(f)

    preprocessed_gaussian_embeddings = np.concatenate([preprocessed_train_gaussian_embeddings, preprocessed_test_gaussian_embeddings],axis=0)
    preprocessed_text_embeddings = np.concatenate([preprocessed_train_text_embeddings, preprocessed_test_text_embeddings],axis=0)

    #########################
    ########################
    preprocessed_gaussian_embeddings/=1000
    #########################
    #########################
    import pdb; pdb.set_trace()

    device = "cuda" if torch.cuda.is_available() else "cpu"

    if args.model_type == 'CLIP':
        criterion = ClipLoss()
    # elif args.model_type == 'MERU':
    else:
        criterion = MeruLoss()
    is_clip_loss = criterion.__class__ in [ClipLoss, MeruLoss]
    loss_specific_kwargs = {
        "logit_scale": 10 if is_clip_loss else np.log(10),
        "logit_bias": None if is_clip_loss else -10,
    }

    recall_fn = partial(recall_n, thresh=0.95, reduce_mean=True)

    print(f"Using device: {device}")
    k_fold = KFold(n_splits=10, shuffle=True)

    metrics = {
        "train": defaultdict(list),
        "validation": defaultdict(list)
    }
    for fold, (train_index, val_index) in enumerate(k_fold.split(preprocessed_text_embeddings)):

        # import pdb; pdb.set_trace()
        train_dataset = TensorDataset(
            torch.from_numpy(preprocessed_gaussian_embeddings[train_index]).float(),
            torch.from_numpy(preprocessed_text_embeddings[train_index]).float(),
        )
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

        val_dataset = TensorDataset(
            torch.from_numpy(preprocessed_gaussian_embeddings[val_index]).float(),
            torch.from_numpy(preprocessed_text_embeddings[val_index]).float(),
        )
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

        if args.model_type == 'CLIP':
            model = ClipModel(
                image_model=nn.Sequential(
                    ResidualHead(output_size, dropout=dropout),
                    ResidualHead(output_size, dropout=dropout),
                    ResidualHead(output_size, dropout=dropout),
                ),
                text_model=nn.Sequential(
                    ProjectionHead(preprocessed_text_embeddings.shape[1], output_size, dropout=dropout),
                    ResidualHead(output_size, dropout=dropout),
                    ResidualHead(output_size, dropout=dropout),
                ),
                **loss_specific_kwargs,
            )
        elif args.model_type == 'MERU':
            model = MeruModel(
                image_model=nn.Sequential(
                    ResidualHead(output_size, dropout=dropout),
                    ResidualHead(output_size, dropout=dropout),
                    ResidualHead(output_size, dropout=dropout),
                ),
                text_model=nn.Sequential(
                    ProjectionHead(preprocessed_text_embeddings.shape[1], output_size, dropout=dropout),
                    ResidualHead(output_size, dropout=dropout),
                    ResidualHead(output_size, dropout=dropout),
                ),
                embed_dim=output_size,
                curv_init=1.0,
                learn_curv=True,
                entail_weight=entail_weight
            )
        elif args.model_type == 'OURS':
            model = OurModel(
                image_model=nn.Sequential(
                    ResidualHead(output_size, dropout=dropout),
                    ResidualHead(output_size, dropout=dropout),
                    nn.Linear(output_size, output_size),
                    nn.LayerNorm(output_size)
                ),
                text_model=nn.Sequential(
                    ProjectionHead(preprocessed_text_embeddings.shape[1], output_size, dropout=dropout),
                ),
                embed_dim=output_size,
                curv_init=1.0,
                learn_curv=True,
                entail_weight=entail_weight
            )

        print(count_parameters(model))
        special_names = ["logit_scale", "visual_alpha", "textual_alpha", "curv"]
        special_params = [p for n, p in model.named_parameters() if n in special_names]
        other_params = [p for n, p in model.named_parameters() if n not in special_names]

        optimizer = torch.optim.AdamW([
                {'params': other_params},
                {'params': special_params, 'weight_decay': 0}
            ],
            lr=lr,
            weight_decay=weight_decay
        )
        scheduler = None  # torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, len(train_loader)*num_epochs)
        output_dir = Path(__file__).parent

        model = model.to(args.device) # cuda 디바이스로 붙인다.

        model, train_loss, val_loss, callback_outputs = train(
            model,
            train_loader=train_loader,
            val_loader=val_loader,
            optimizer=optimizer,
            scheduler=scheduler,
            criterion=criterion,
            num_epochs=num_epochs,
            device=device,
            verbose=True,
            output_dir=output_dir,
            # clip_grad_norm=0.3,
            callbacks=[
                # You can comment those callbacks to fasten the training
                # they are here to help understand what is happening across epochs
                # recall_n_callback(val_loader, n=10, device=device),
                # diagonal_callback(val_loader, device=device),
                # non_diagonal_callback(val_loader, device=device),
                # check_model_parameter_callback("logit_scale"),
                # check_model_parameter_callback("logit_bias"),
            ],
            save_idx=args.save_idx
        )

        if plot_verbose:
            callback_plot_kwargs = [
                {"ylabel": "Validation\nRecall@10", "color": "b", "ylim": [0, 1]},
                {"ylabel": "Diagonal Mean", "color": "b", "ylim": [1e-7, 1], "yscale": "log"},
                {"ylabel": "Non-diagonal Mean", "color": "b", "ylim": [1e-7, 1], "yscale": "log"},
                {"ylabel": "Logit scale", "color": "black"},
                {"ylabel": "Logit bias", "color": "black"},
            ]
            plot_training(
                train_loss,
                val_loss,
                callback_outputs,
                callback_kwargs=callback_plot_kwargs,
            )

        # Define a small train dataset to get metrics faster
        small_train_dataset = TensorDataset(
            torch.from_numpy(preprocessed_gaussian_embeddings[train_index][:1000]).float(),
            torch.from_numpy(preprocessed_text_embeddings[train_index][:1000]).float(),
        )
        small_train_loader = DataLoader(small_train_dataset, batch_size=batch_size, shuffle=False)
        for loader_name, loader, weights_path in [
            ("train", small_train_loader, output_dir / f"last_{args.save_idx}.pt"),
            ("validation", val_loader, output_dir / f"best_val_{args.save_idx}.pt"),
        ]:
            model.load_state_dict(torch.load(weights_path, weights_only=True))
            image_embeddings, text_embeddings = predict(model, loader, device=device)
            _curv = model.curv.exp().cpu()
            similarity = -torch.abs(L.pairwise_oxy_angle(text_embeddings, image_embeddings, _curv)).detach().numpy()
            similarity = similarity.T


            if plot_verbose:
                # Plot similarity matrices that should be diagonal
                fig, axes = plt.subplots(nrows=1, ncols=3, figsize=(10, 5))
                gauss_similarity = (image_embeddings @ image_embeddings.T).numpy()
                plot_matrix(gauss_similarity[:100, :100], ax=axes[0], title="Gauss-to-Gauss")
                text_similarity = (text_embeddings @ text_embeddings.T).numpy()
                plot_matrix(text_similarity[:100, :100], ax=axes[1], title="Text-to-text")
                plot_matrix(similarity[:100, :100], ax=axes[2], title="Gauss-to-Text")
                fig.suptitle(f"Learnt similarities - {loader_name}")
                plt.tight_layout()
                plt.show()
                plt.close()

            random_perf = 10 / len(similarity)

            nq_perf_5 = recall_fn(similarity, np.eye(len(similarity)), n_first=5)
            nq_perf_10 = recall_fn(similarity, np.eye(len(similarity)), n_first=10)
            nq_perf_100 = recall_fn(similarity, np.eye(len(similarity)), n_first=100)

            metrics[loader_name]["recall@5"].append(100*nq_perf_5)
            metrics[loader_name]["recall@10"].append(100*nq_perf_10)
            metrics[loader_name]["recall@100"].append(100*nq_perf_100)


        print(f"Metrics after {fold+1} folds")
        for loader_name in ["train", "validation"]:
            print("="*10, loader_name, "="*10)
            for metric_name in ["recall@5", "recall@10", "recall@100"]:
                print(f"{metric_name}: {np.mean(metrics[loader_name][metric_name]):.3f} +- {np.std(metrics[loader_name][metric_name]):.3f}")

    os.remove(output_dir / f"last_{args.save_idx}.pt")
    os.remove(output_dir / f"best_val_{args.save_idx}.pt")

