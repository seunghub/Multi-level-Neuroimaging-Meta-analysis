import numpy as np
import torch
import tqdm
import torch.nn.functional as F


def mean_pooling(token_embeddings, attention_mask):
    input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)


def embed(input_ids, attention_mask, model, device=torch.device("cpu")):
    with torch.no_grad():
        input_ids = input_ids.to(device)
        attention_mask = attention_mask.to(device)

        out = model(
            input_ids,
            attention_mask=attention_mask,
            output_hidden_states=True,
        )
        # Perform pooling
        text_embeddings = mean_pooling(out.hidden_states[-1], attention_mask)

        # Normalize embeddings
        text_embeddings = F.normalize(text_embeddings, p=2, dim=1)

    text_embeddings = text_embeddings.cpu()

    return text_embeddings.numpy()


def embed_texts(texts, tokenizer, model, device=torch.device("cpu")):
    tokenized = tokenizer(
        texts,
        truncation=True,
        padding=True,
        return_tensors="pt",
    )

    return embed(tokenized.input_ids, tokenized.attention_mask, model, device)


def batch_embed_texts(texts, tokenizer, model, batch_size=8):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    return np.concatenate([
        embed_texts(texts[index:index + batch_size], tokenizer, model, device)
        for index in tqdm.trange(0, len(texts), batch_size)
    ], axis=0)


def chunk_tokenize(text, text_id, tokenizer, model_max_length):
    """
    Tokenize a text and split it into chunks of size `model_max_length`
    """
    tokenized = tokenizer(text, return_tensors="pt",)
    input_ids = tokenized["input_ids"]
    attention_mask = tokenized["attention_mask"]

    chunk_input_ids = [
        input_ids[0, i:i + model_max_length]
        for i in range(0, input_ids.shape[1], model_max_length)
    ]
    
    for i in range(0, input_ids.shape[1], model_max_length):
        print(f"i: {i}")
        print(f"input_ids[0,:]: {input_ids[0,:]}")
        print(f"[0, i:i + model_max_length]: {input_ids[0, i:i + model_max_length]}")
    
    text_id = [text_id] * len(chunk_input_ids)
    chunk_attention_mask = [
        attention_mask[0, i:i + model_max_length]
        for i in range(0, attention_mask.shape[1], model_max_length)
    ]
    last_chunk = tokenizer(
        tokenizer.decode(chunk_input_ids[-1]),
        padding="max_length",
        max_length=model_max_length,
        return_tensors="pt",
    )
    chunk_input_ids[-1] = last_chunk["input_ids"][0]
    chunk_attention_mask[-1] = last_chunk["attention_mask"][0]
    chunk_input_ids = torch.cat([torch.unsqueeze(el, dim=0) for el in chunk_input_ids], dim=0)
    chunk_attention_mask = torch.cat([torch.unsqueeze(el, dim=0) for el in chunk_attention_mask], dim=0)

    return chunk_input_ids, chunk_attention_mask, text_id

def chunk_tokenize_texts(texts, text_ids, tokenizer, model_max_length):
    """
    Tokenize a list of texts and split them into chunks of size `model_max_length`
    """
    tokenized = [
        chunk_tokenize(text, text_id, tokenizer, model_max_length)
        for text, text_id in tqdm.tqdm(zip(texts, text_ids), total=len(texts))
    ]
    input_ids = torch.cat([el[0] for el in tokenized], dim=0)
    attention_masks = torch.cat([el[1] for el in tokenized], dim=0)
    text_ids = [
        text_id
        for _, _, chunk_text_ids in tokenized
        for text_id in chunk_text_ids
    ]

    return input_ids, attention_masks, text_ids



def batch_embed_texts_by_chunks(texts, text_ids, tokenizer, model, model_max_length, batch_size=8):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    input_ids, attention_masks, text_ids = chunk_tokenize_texts(
        texts,
        text_ids,
        tokenizer,
        model_max_length,
    )

    return np.concatenate([
        embed(
            input_ids[index:index + batch_size],
            attention_masks[index:index+batch_size],
            model,
            device,
        )
        for index in tqdm.trange(0, len(input_ids), batch_size)
    ], axis=0), text_ids
