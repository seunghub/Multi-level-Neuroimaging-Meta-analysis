import math
import matplotlib.pyplot as plt
import numpy as np
from collections import Counter, defaultdict
from random import choice
import torch
import torch.nn as nn
import tqdm
from torch.utils.data import DataLoader, TensorDataset
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LambdaLR



def recall_n(y_pred, y_truth, n_first=10, thresh=0.95, reduce_mean=False):
    assert (y_pred.ndim in (1, 2)) and (
        y_truth.ndim in (1, 2)
    ), "arrays should be of dimension 1 or 2"
    assert y_pred.shape == y_truth.shape, "both arrays should have the same shape"

    if y_pred.ndim == 1:
        # recall@n for a single sample
        targets = np.where(y_truth >= thresh)[0]
        pred_n_first = np.argsort(y_pred)[::-1][:n_first]

        if len(targets) > 0:
            ratio_in_n = len(np.intersect1d(targets, pred_n_first)) / len(targets)
        else:
            ratio_in_n = np.nan

        return ratio_in_n
    else:
        # recall@n for a dataset (mean of recall@n for all samples)
        result = np.zeros(len(y_pred))
        for i, (sample_y_pred, sample_y_truth) in enumerate(zip(y_pred, y_truth)):
            result[i] = recall_n(sample_y_pred, sample_y_truth, n_first, thresh)
        if reduce_mean:
            return np.nanmean(result)

        return result


def sigmoid(x):
    return 1/(1 + np.exp(-x))

def get_cosine_schedule_with_warmup(
    optimizer: Optimizer, num_warmup_steps: int, num_training_steps: int, num_cycles: float = 0.5, last_epoch: int = -1
):
    """
    Create a schedule with a learning rate that decreases following the values of the cosine function between the
    initial lr set in the optimizer to 0, after a warmup period during which it increases linearly between 0 and the
    initial lr set in the optimizer.

    Args:
        optimizer (:class:`~torch.optim.Optimizer`):
            The optimizer for which to schedule the learning rate.
        num_warmup_steps (:obj:`int`):
            The number of steps for the warmup phase.
        num_training_steps (:obj:`int`):
            The total number of training steps.
        num_cycles (:obj:`float`, `optional`, defaults to 0.5):
            The number of waves in the cosine schedule (the defaults is to just decrease from the max value to 0
            following a half-cosine).
        last_epoch (:obj:`int`, `optional`, defaults to -1):
            The index of the last epoch when resuming training.

    Return:
        :obj:`torch.optim.lr_scheduler.LambdaLR` with the appropriate schedule.
    """

    def lr_lambda(current_step):
        if current_step < num_warmup_steps:
            return float(current_step) / float(max(1, num_warmup_steps))
        progress = float(current_step - num_warmup_steps) / float(max(1, num_training_steps - num_warmup_steps))
        return max(0.0, 0.5 * (1.0 + math.cos(math.pi * float(num_cycles) * 2.0 * progress)))

    return LambdaLR(optimizer, lr_lambda, last_epoch)


def plot_training(clip_train_loss, clip_val_loss, callback_outputs=None, callback_kwargs=None):
    fontsize=14
    callback_outputs = callback_outputs if callback_outputs else []
    num_callbacks = len(callback_outputs[0]) if callback_outputs else 0
    num_epochs = len(clip_train_loss)
    fig, axes = plt.subplots(nrows=2+num_callbacks, ncols=1, sharex=True, figsize=(10, 4+num_callbacks*2))
    axes[0].plot(
        range(num_epochs),
        clip_train_loss,
        linestyle="-",
        markersize=3,
        color="r",
        label="CLIP - Training",
    )
    axes[0].plot(
        range(num_epochs),
        clip_val_loss,
        linestyle="-",
        markersize=3,
        color="b",
        label="CLIP - Validation",
    )
    axes[0].set_yscale("log")
    # axes[0].set_xticks([])
    # axes[0].set_xlabel("Epochs")
    axes[0].set_ylabel("CLIP Loss", fontsize=fontsize)
    axes[1].plot(
        range(num_epochs),
        clip_train_loss,
        linestyle="-",
        markersize=3,
        color="r",
        label="CLIP - Training",
    )
    axes[1].plot(
        range(num_epochs),
        clip_val_loss,
        linestyle="-",
        markersize=3,
        color="b",
        label="CLIP - Validation",
    )
    # axes[1].set_xticks(list(range(0, num_epochs, int(num_epochs / 10))))
    # axes[1].set_xlabel("Epochs")
    axes[1].set_ylabel("CLIP Loss", fontsize=fontsize)
    plt.legend()
    if callback_outputs:
        for index in range(num_callbacks):
            axes[2 + index].plot(
                range(num_epochs),
                [callback_outputs[epoch_index][index] for epoch_index in range(num_epochs)],
                linestyle="-",
                markersize=3,
                color=callback_kwargs[index]["color"] if "color" in callback_kwargs[index] else "r",
            )
            axes[2 + index].set_xticks(list(range(0, num_epochs, int(num_epochs / 10))))
            if "xlabel" in callback_kwargs[index]:
                axes[2 + index].set_xlabel(callback_kwargs[index]["xlabel"], fontsize=fontsize)
            if "ylabel" in callback_kwargs[index]:
                axes[2 + index].set_ylabel(callback_kwargs[index]["ylabel"], fontsize=fontsize)
            if "xlim" in callback_kwargs[index]:
                axes[2 + index].set_xlim(callback_kwargs[index]["xlim"])
            if "ylim" in callback_kwargs[index]:
                axes[2 + index].set_ylim(callback_kwargs[index]["ylim"])
            if "yscale" in callback_kwargs[index]:
                axes[2 + index].set_yscale(callback_kwargs[index]["yscale"])
    plt.tight_layout()
    plt.show()
    plt.close()


def clip_predict_embeddings(model, data_loader, num_samples=None, device="cpu"):
    model.eval()

    num_predicted_samples = 0
    with torch.no_grad():
        all_image_features = []
        all_text_features = []
        for _, batch in enumerate(data_loader):
            nv_gaussian, nv_difumo = (
                batch[0].to(device),
                batch[1].to(device),
            )
            peak_output, text_output, *_ = model(nv_gaussian, nv_difumo)

            image_features = peak_output / peak_output.norm(dim=-1, keepdim=True)
            text_features = text_output / text_output.norm(dim=-1, keepdim=True)
            image_features = image_features.cpu().numpy()
            all_image_features.append(image_features)
            text_features = text_features.cpu().numpy()
            all_text_features.append(text_features)

            num_predicted_samples += len(image_features)
            if num_samples is not None and num_predicted_samples >= num_samples:
                break

    all_image_features = np.concatenate(all_image_features, axis=0)
    all_text_features = np.concatenate(all_text_features, axis=0)

    return all_image_features, all_text_features

def clip_predict(model, data_loader, num_samples=None, device="cpu"):
    """
    A particularity of the prediction for CLIP is that both batch[0] and batch[1]
    are used and passed to their respective encoder.
    """
    all_image_features, all_text_features = clip_predict_embeddings(
        model, data_loader, num_samples=num_samples, device=device,
    )
    text_probs = sigmoid((100.0 * all_image_features @ all_text_features.T))

    return text_probs



def predict(model, val_loader, device="cpu", verbose=False):
    model.eval()

    outputs = []
    with torch.no_grad():
        for _, batch in tqdm.tqdm(
            enumerate(val_loader),
            total=len(val_loader),
            disable=not verbose,
        ):
            data, _ = (
                batch[0].to(device),
                batch[1].to(device),
            )
            output = model(data)
            outputs.append(output.detach().cpu().numpy())

    return np.concatenate(outputs, axis=0)


def val(model, val_loader, criterion, device="cpu"):
    model.eval()

    loss_values = []
    with torch.no_grad():
        for _, batch in enumerate(val_loader, 0):
            data, target = (
                batch[0].to(device),
                batch[1].to(device),
            )
            output = model(data)
            loss = criterion(output, target)
            loss_values.append(loss.item())
    return np.mean(loss_values)


def train(
    model,
    train_loader,
    val_loader,
    optimizer,
    criterion,
    num_epochs=100,
    steps_per_epoch=None,
    device="cpu",
    verbose=False,
):
    loss_train = []
    loss_val = []
    best_state_dict = None
    best_val_loss = None

    model = model.to(device)
    for epoch_index in tqdm.trange(num_epochs, disable=not verbose):
        number_of_steps = 0

        model.train()

        batch_loss = []

        for batch in train_loader:
            number_of_steps += len(batch[0])

            # Retrieve mini-batch
            data, target = (
                batch[0].to(device),
                batch[1].to(device),
            )
            # Forward pass
            output = model(data)
            # Loss computation
            loss = criterion(output, target)
            batch_loss.append(loss.item())
            # Backpropagation (gradient computation)
            loss.backward()
            # Parameter update
            optimizer.step()
            # Erase previous gradients
            optimizer.zero_grad()

            if (steps_per_epoch is not None) and (number_of_steps > steps_per_epoch):
                break

        # Compute mean epoch loss over all batches
        epoch_train_loss = np.mean(batch_loss)
        loss_train.append(epoch_train_loss)
        # Compute val loss
        epoch_val_loss = val(model, val_loader, criterion, device=device)

        if best_val_loss is None or epoch_val_loss < best_val_loss:
            best_state_dict = model.state_dict()

        loss_val.append(epoch_val_loss)

    model.load_state_dict(best_state_dict)

    return model, loss_train, loss_val


def mlp_training(
    model,
    X_train,
    X_test,
    y_train,
    y_test,
    criterion=nn.MSELoss(),
    batch_size=32,
    num_epochs=100,
    verbose=True,
    plot_loss=True,
):
    train_loader = DataLoader(
        TensorDataset(torch.from_numpy(X_train).float(), torch.from_numpy(y_train).float()),
        shuffle=True,
        batch_size=batch_size,
    )
    val_loader = DataLoader(
        TensorDataset(torch.from_numpy(X_test).float(), torch.from_numpy(y_test).float()),
        batch_size=batch_size,
    )
    optimizer = torch.optim.AdamW(model.parameters())

    device = "cuda" if torch.cuda.is_available() else "cpu"

    model, loss_train, loss_val = train(
        model,
        train_loader,
        val_loader,
        optimizer,
        criterion,
        num_epochs=num_epochs,
        verbose=verbose,
        device=device,
    )
    if plot_loss:
        import matplotlib.pyplot as plt
        plt.figure()
        plt.plot()
        plt.plot(list(range(len(loss_train))), loss_train, label="train")
        plt.plot(list(range(len(loss_val))), loss_val, label="val")
        plt.title("Loss")
        plt.legend(loc="upper right")
        plt.show()
    y_pred = predict(model, val_loader, device)

    return model, y_pred

def group_embeddings_by_chunks(train_text_embeddings, pmid):
    grouped_embeddings = defaultdict(list)
    
    for embedding, chunk_id in zip(train_text_embeddings, pmid):
        grouped_embeddings[chunk_id].append(embedding)
    
    # Convert grouped embeddings to a list of numpy arrays
    chunked_embeddings = [np.stack(embeddings) for embeddings in grouped_embeddings.values()]
    
    return chunked_embeddings


def pad_and_mask(chunked_embeddings, max_chunks):
    embed_dim = chunked_embeddings[0].shape[1]
    
    padded_embeddings = []
    masks = []
    
    for chunks in chunked_embeddings:
        padding_length = max_chunks - len(chunks)
        padding = np.zeros((padding_length, embed_dim))
        padded_chunks = np.vstack([chunks, padding])
        padded_embeddings.append(padded_chunks)
        
        mask = [1] * len(chunks) + [0] * padding_length
        masks.append(mask)
    
    padded_embeddings = np.stack(padded_embeddings)
    masks = np.array(masks, dtype=np.bool_)
    
    return padded_embeddings, masks
