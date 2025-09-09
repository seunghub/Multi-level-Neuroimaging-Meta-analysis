import numpy as np
import torch
import torch.nn as nn
import tqdm
from src.utils import plot_training, recall_n
from functools import partial
import lorentz as L
from copy import deepcopy

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def predict(model, data_loader, device="cpu"):
    model.eval()

    with torch.no_grad():
        all_image_embeddings = []
        all_text_embeddings = []
        for _, batch in enumerate(data_loader):
            image, text = (
                batch[0].to(device),
                batch[1].to(device),
            )
            pred = model(image, text)
            image_embeddings, text_embeddings = pred['image_embedding'], pred['text_embedding']
            image_embeddings = image_embeddings.cpu()
            all_image_embeddings.append(image_embeddings)
            text_embeddings = text_embeddings.cpu()
            all_text_embeddings.append(text_embeddings)

    all_image_embeddings = torch.cat(all_image_embeddings, dim=0)
    all_text_embeddings = torch.cat(all_text_embeddings, dim=0)

    return all_image_embeddings, all_text_embeddings

def predict_autoencoder(model, data_loader, device="cpu"):
    model.eval()

    with torch.no_grad():
        all_image_embeddings = []
        all_text_embeddings = []
        all_latent_decoded = []
        for _, batch in enumerate(data_loader):
            image, text = (
                batch[0].to(device),
                batch[1].to(device),
            )
            image_embeddings, text_embeddings, latent_decoded = model(image, text)
            image_embeddings = image_embeddings.cpu()
            all_image_embeddings.append(image_embeddings)
            text_embeddings = text_embeddings.cpu()
            all_text_embeddings.append(text_embeddings)
            latent_decoded = latent_decoded.cpu()
            all_latent_decoded.append(latent_decoded)

    all_image_embeddings = torch.cat(all_image_embeddings, dim=0)
    all_text_embeddings = torch.cat(all_text_embeddings, dim=0)
    all_latent_decoded = torch.cat(all_latent_decoded, dim=0)

    return all_image_embeddings, all_text_embeddings, all_latent_decoded

def val(model, val_loader, criterion, device="cpu"):
    model.eval()
    recall_fn = partial(recall_n, thresh=0.95, reduce_mean=True)
    images = []
    texts = []
    with torch.no_grad():
        for _, batch in enumerate(val_loader, 0):
            image, text = (
                batch[0].to(device),
                batch[1].to(device),
            )
            images.append(image)
            texts.append(text)
    images = torch.cat(images,dim=0)
    texts = torch.cat(texts,dim=0)

    pred = model(images, texts)
    # image_embeddings = pred['image_embedding']
    # text_embeddings = pred['text_embedding']

    similarity = -torch.abs(L.pairwise_oxy_angle(pred['image_embedding'], pred['text_embedding'], model.curv.exp())).cpu().detach().numpy()
    nq_perf = recall_fn(similarity, np.eye(len(similarity)), n_first=10)
    return nq_perf

def val_autoencoder(model, val_loader, criterion, criterion_mse, beta, alpha, device="cpu"):
    model.eval()

    val_batch = []
    val_contrastive_batch = []
    val_mse_batch = []
    with torch.no_grad():
        for _, batch in enumerate(val_loader, 0):
            image_embeddings, text_embeddings = (
                batch[0].to(device),
                batch[1].to(device),
            )
            image_embed, text_embed, latent_decoded = model(image_embeddings, text_embeddings)
            
            bs = image_embeddings.size(0)
            contrastive_loss = criterion(image_embed, text_embed, model.logit_scale, model.logit_bias)
            mse_loss = criterion_mse(image_embeddings, latent_decoded)
            min_value = torch.min(torch.cat((image_embeddings, latent_decoded)))
            max_value = torch.max(torch.cat((image_embeddings, latent_decoded)))
            range_value = max_value - min_value
            scaling_factor_MinMax = 1 / range_value
            scaling_factor_Max = 1 / max_value
            image_embeddings_norm = torch.norm(image_embeddings, p='fro')
            latent_decoded_norm = torch.norm(latent_decoded, p='fro')
            scaling_factor_norm = 1 / (image_embeddings_norm * latent_decoded_norm)
            scaling_factor_cotrs = contrastive_loss.item() / mse_loss.item()
            scaling_factor_off = 1
            scaled_mse_loss = mse_loss * scaling_factor_off
            loss = alpha * contrastive_loss + beta * scaled_mse_loss

            val_batch.append(loss.item())
            val_contrastive_batch.append(contrastive_loss.item())
            val_mse_batch.append(scaled_mse_loss.item())
    return np.mean(val_batch), np.mean(val_contrastive_batch), np.mean(val_mse_batch)

def train(model, train_loader, val_loader, optimizer, criterion,
          scheduler=None, num_epochs=100, device="cpu", verbose=False,
          output_dir=None, callbacks=None, clip_grad_norm=None, save_idx=0):
    loss_train = []
    loss_val = []
    best_state_dict = None
    best_val_recall = None
    callbacks_outputs = []
    model = model.to(device)

    for epoch_index in tqdm.trange(num_epochs, disable=not verbose):
        model.train()

        batch_loss = []

        for batch in train_loader:
            # Erase previous gradients
            optimizer.zero_grad()
            # Retrieve mini-batch
            # image_embeddings, text_embeddings, mask = (
            #     batch[0].to(device),
            #     batch[1].to(device),
            #     batch[2].to(device),
            # )

            image_embeddings, text_embeddings = (
                batch[0].to(device),
                batch[1].to(device),
                # batch[2].to(device),
            )

            # print(f"extract input data")
            # Forward pass
            # print(f"model: {model}")
            pred = model(image_embeddings, text_embeddings)
            # print(model.curv.exp())
            image_embed, text_embed, entailment_loss, contrastive_loss =\
                 pred['image_embedding'], pred['text_embedding'], pred['entailment_loss'], pred['contrastive_loss']

            # Loss computation
            # loss = criterion(image_embed, text_embed, model.logit_scale, model.curv.exp(), model.entail_weight * model.entailment_loss)
            loss = entailment_loss + contrastive_loss
            batch_loss.append(loss.item())

            # Backpropagation (gradient computation)
            loss.backward()

            if clip_grad_norm is not None:
                torch.nn.utils.clip_grad_norm_(model.parameters(), clip_grad_norm)

            # Parameter update
            optimizer.step()
            # Scheduler update
            if scheduler is not None:
                scheduler.step()

        # Compute mean epoch loss over all batches
        epoch_train_loss = np.mean(batch_loss)
        if np.isnan(epoch_train_loss):
            raise ValueError("Training loss is NaN. Consider decreasing the learning rate.")
        loss_train.append(epoch_train_loss)

        # Compute val loss
        epoch_val_recall = val(model, val_loader, criterion, device=device)
        if callbacks is not None:
            callbacks_outputs.append([
                callback(model, epoch_index) for callback in callbacks
            ])

        if best_val_recall is None or epoch_val_recall > best_val_recall:
            # if best_val_recall is not None: print(f'updataed!! from {best_val_recall:.3f} to {epoch_val_recall:.3f}')
            best_val_recall = epoch_val_recall
            best_state_dict = deepcopy(model.state_dict())
        loss_val.append(epoch_val_recall)

        # print(model.curv.exp().cpu().item())
        # import pdb; pdb.set_trace()
    if output_dir is not None:
        torch.save(model.state_dict(), output_dir / f"last_{save_idx}.pt")
        torch.save(best_state_dict, output_dir / f"best_val_{save_idx}.pt")
    
    model.load_state_dict(best_state_dict)
    if callbacks is not None:
        return model, loss_train, loss_val, callbacks_outputs

    return model, loss_train, loss_val


def train_autoencoder(model, train_loader, val_loader, optimizer_encoder, optimizer_decoder, criterion, beta, alpha,
          scheduler=None, num_epochs=100, device="cpu", verbose=False,
          output_dir=None, callbacks=None, clip_grad_norm=None):
    loss_train = []
    loss_val = []
    loss_contrastive_train = []
    loss_contrastive_val = []
    loss_mse_train = []
    loss_mse_val = []
    best_state_dict = None
    best_val_loss = None
    callbacks_outputs = []
    model = model.to(device)

    for epoch_index in tqdm.trange(num_epochs, disable=not verbose):
        model.train()

        batch_loss = []
        batch_loss_contrastive = []
        batch_loss_mse = []

        for batch in train_loader:
            # Erase previous gradients
            optimizer_encoder.zero_grad()
            optimizer_decoder.zero_grad()
            # Retrieve mini-batch
            # image_embeddings, text_embeddings, mask = (
            #     batch[0].to(device),
            #     batch[1].to(device),
            #     batch[2].to(device),
            # )

            image_embeddings, text_embeddings = (
                batch[0].to(device),
                batch[1].to(device),
                # batch[2].to(device),
            )

            # print(f"extract input data")
            # Forward pass
            # print(f"model: {model}")
            image_embed, text_embed, latent_decoded = model(image_embeddings, text_embeddings)

            # Loss computation
            criterion_mse = nn.MSELoss()
            bs = image_embeddings.size(0)
            contrastive_loss = criterion(image_embed, text_embed, model.logit_scale, model.logit_bias)
            mse_loss = criterion_mse(image_embeddings, latent_decoded) 
            
            # Compute the range of embeddings
            min_value = torch.min(torch.cat((image_embeddings, latent_decoded)))
            max_value = torch.max(torch.cat((image_embeddings, latent_decoded)))
            range_value = max_value - min_value
            scaling_factor_MinMax = 1 / range_value
            scaling_factor_Max = 1 / max_value
            image_embeddings_norm = torch.norm(image_embeddings, p='fro')
            latent_decoded_norm = torch.norm(latent_decoded, p='fro') 
            scaling_factor_norm = 1 / (image_embeddings_norm * latent_decoded_norm)
            scaling_factor_cotrs = contrastive_loss.item() / mse_loss.item()
            scaling_factor_off = 1
            scaled_mse_loss = mse_loss * scaling_factor_off
            loss = alpha * contrastive_loss + beta * scaled_mse_loss
            # loss = alpha * criterion(image_embed, text_embed, model.logit_scale, model.logit_bias) + beta * criterion_mse(image_embeddings, latent_decoded) / bs
            
            batch_loss.append(loss.item())
            batch_loss_contrastive.append(contrastive_loss.item())
            batch_loss_mse.append(scaled_mse_loss.item())

            # Backpropagation (gradient computation)
            loss.backward()

            if clip_grad_norm is not None:
                torch.nn.utils.clip_grad_norm_(model.parameters(), clip_grad_norm)

            # Parameter update
            optimizer_encoder.step()
            optimizer_decoder.step()
            # Scheduler update
            if scheduler is not None:
                scheduler.step()

        # Compute mean epoch loss over all batches
        epoch_train_loss = np.mean(batch_loss)
        epoch_train_loss_contrastive = np.mean(batch_loss_contrastive)
        epoch_train_loss_mse = np.mean(batch_loss_mse)
        if np.isnan(epoch_train_loss):
            raise ValueError("Training loss is NaN. Consider decreasing the learning rate.")
        loss_train.append(epoch_train_loss)
        loss_contrastive_train.append(epoch_train_loss_contrastive)
        loss_mse_train.append(epoch_train_loss_mse)

        # Compute val loss
        epoch_val_loss, epoch_val_loss_contrastive, epoch_val_loss_mse = val_autoencoder(
            model, val_loader, 
            criterion, 
            criterion_mse, 
            beta, alpha, device=device)

        if callbacks is not None:
            callbacks_outputs.append([
                callback(model, epoch_index) for callback in callbacks
            ])

        if best_val_loss is None or epoch_val_loss < best_val_loss:
            best_val_loss = epoch_val_loss
            best_state_dict = model.state_dict()
        loss_val.append(epoch_val_loss)
        loss_contrastive_val.append(epoch_val_loss_contrastive)
        loss_mse_val.append(epoch_val_loss_mse)

    if output_dir is not None:
        torch.save(best_state_dict, output_dir / "best_val.pt")
        torch.save(model.state_dict(), output_dir / "last.pt")

    model.load_state_dict(best_state_dict)
    if callbacks is not None:
        return model, loss_train, loss_val, loss_contrastive_train, loss_contrastive_val, loss_mse_train, loss_mse_val, callbacks_outputs

    return model, loss_train, loss_val, loss_contrastive_train, loss_contrastive_val, loss_mse_train, loss_mse_val


def single_input_predict(model, data_loader, device="cpu"):
    model.eval()

    with torch.no_grad():
        all_embeddings = []
        for _, batch in enumerate(data_loader):
            inputs = batch[0].to(device)
            preds = model(inputs)

            preds = preds.cpu()
            all_embeddings.append(preds)

    all_embeddings = torch.cat(all_embeddings, dim=0)

    return all_embeddings


def single_input_val(model, val_loader, criterion, device="cpu"):
    model.eval()

    val_batch = []
    with torch.no_grad():
        for _, batch in enumerate(val_loader, 0):
            inputs, targets = (
                batch[0].to(device),
                batch[1].to(device),
            )
            preds = model(inputs)

            loss = criterion(
                preds,
                targets,
            )
            val_batch.append(loss.item())
    return np.mean(val_batch)


def single_input_train(model, train_loader, val_loader, optimizer, criterion,
          scheduler=None, num_epochs=100, device="cpu", verbose=False, output_dir=None):
    loss_train = []
    loss_val = []
    best_state_dict = None
    best_val_loss = None
    model = model.to(device)

    for epoch_index in tqdm.trange(num_epochs, disable=not verbose):
        model.train()

        batch_loss = []

        for batch in train_loader:
            # Erase previous gradients
            optimizer.zero_grad()
            # Retrieve mini-batch
            inputs, targets = (
                batch[0].to(device),
                batch[1].to(device),
            )
            # Forward pass
            preds = model(inputs)

            # Loss computation
            loss = criterion(preds, targets)
            batch_loss.append(loss.item())

            # Backpropagation (gradient computation)
            loss.backward()

            # Parameter update
            optimizer.step()
            # Scheduler update
            if scheduler is not None:
                scheduler.step()

        # Compute mean epoch loss over all batches
        epoch_train_loss = np.mean(batch_loss)
        if np.isnan(epoch_train_loss):
            raise ValueError("Training loss is NaN. Consider decreasing the learning rate.")
        loss_train.append(epoch_train_loss)

        # Compute val loss
        epoch_val_loss = single_input_val(model, val_loader, criterion, device=device)

        if best_val_loss is None or epoch_val_loss < best_val_loss:
            best_val_loss = epoch_val_loss
            best_state_dict = model.state_dict()
            if output_dir is not None:
                torch.save(best_state_dict, output_dir / "best_val.pt")

        loss_val.append(epoch_val_loss)

    if output_dir is not None:
        torch.save(model.state_dict(), output_dir / "last.pt")

    model.load_state_dict(best_state_dict)

    return model, loss_train, loss_val

def recall_n_callback(loader, n, device):
    def run_callback(model, epoch_index):
        model.eval()

        with torch.no_grad():
            image_embeddings, text_embeddings = predict(
                model,
                loader,
                device=device,
            )
            similarity = (image_embeddings @ text_embeddings.T).softmax(dim=1).cpu().numpy()

        recall = recall_n(
            similarity,
            np.eye(len(similarity)),
            n_first=n,
            thresh=0.95,
            reduce_mean=True,
        )
        return recall

    return run_callback

def diagonal_callback(loader, device):
    def run_callback(model, epoch_index):
        model.eval()

        with torch.no_grad():
            image_embeddings, text_embeddings = predict(
                model,
                loader,
                device=device,
            )
            similarity = (image_embeddings @ text_embeddings.T).softmax(dim=1).cpu().numpy()

        return np.mean(np.diag(similarity))

    return run_callback

def non_diagonal_callback(loader, device):
    def run_callback(model, epoch_index):
        model.eval()

        with torch.no_grad():
            image_embeddings, text_embeddings = predict(
                model,
                loader,
                device=device,
            )
            similarity = (image_embeddings @ text_embeddings.T).softmax(dim=1).cpu().numpy()

        return np.mean(similarity - np.diag(np.diag(similarity)))

    return run_callback

def check_model_parameter_callback(attribute_name):
    def run_callback(model, epoch_index):
        return getattr(model, attribute_name).detach().cpu().numpy()

    return run_callback
