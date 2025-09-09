import pickle
from pathlib import Path

import numpy as np
import torch
import tqdm
from sklearn.model_selection import train_test_split


def get_sample_from_publication(data_pub, tokenizer, l_seq):
    token_ids = (
            [tokenizer.bos_token_id]
            + data_pub["token_ids"]["body"]
            + data_pub["token_ids"]["abstract"]
            + [tokenizer.eos_token_id]
    )
    n_tokens = len(token_ids) - l_seq
    if n_tokens <= 0:
        input_ids = (
            token_ids + [tokenizer.eos_token_id] * (-n_tokens)
        )
        attention_masks = (
            [1] * len(token_ids) + [0] * (-n_tokens)
        )
    else:
        seq_id = np.random.randint(n_tokens)
        input_ids = (token_ids[seq_id:seq_id + l_seq])
        attention_masks = ([1] * l_seq)

    return input_ids, attention_masks


class ChunkDataset(torch.utils.data.Dataset):
    def __init__(self, data_path, tokenizer, max_number_of_chunks=None, l_seq=None):
        super().__init__()

        self.tokenizer = tokenizer
        self.l_seq = l_seq if l_seq is not None else tokenizer.model_max_length

        if isinstance(data_path, str) or isinstance(data_path, Path):
            self.list_files = list(Path(data_path).glob("*.pkl"))[:-1]
        elif isinstance(data_path, list):
            self.list_files = data_path

        if max_number_of_chunks is not None:
            self.list_files = self.list_files[:max_number_of_chunks]

        self.n_files = len(self.list_files)


class DiskDataset(ChunkDataset):
    def __init__(self, data_path, tokenizer, max_number_of_chunks=None,
                 l_seq=None, num_samples_per_pub=1000, batch_size=8):
        super().__init__(data_path, tokenizer, max_number_of_chunks, l_seq)

        # Length of the sequences
        self.batch_size = batch_size
        self.n_chunks = int(num_samples_per_pub / self.batch_size)
        self.n_samples = self.n_files * self.n_chunks

    def __len__(self):
        return self.n_samples

    def __getitem__(self, idx):
        idx_file = idx // self.n_chunks
        idx_entry = idx % self.n_chunks
        with open(self.list_files[idx_file], "rb") as f:
            data_pubs = pickle.load(f)

        input_ids = []
        attention_masks = []
        for pub_idx in range(self.batch_size):
            pub_input_ids, pub_attention_masks = get_sample_from_publication(
                data_pubs[idx_entry + pub_idx],
                self.tokenizer,
                self.l_seq,
            )
            input_ids.append(pub_input_ids)
            attention_masks.append(pub_attention_masks)

        input_ids = torch.tensor(input_ids)
        attention_masks = torch.tensor(attention_masks)
        return dict(
            input_ids=input_ids,
            labels=input_ids,
            attention_mask=attention_masks,
        )


class InMemoryDataset(ChunkDataset):
    def __init__(self, data_path, tokenizer, max_number_of_chunks=None, l_seq=None):
        super().__init__(data_path, tokenizer, max_number_of_chunks, l_seq)

        self.data_files = [
            pickle.load(open(file_path, "rb"))
            for file_path in tqdm.tqdm(self.list_files, total=len(self.list_files))
        ]
        self.data_files = [
            pub
            for pubs in self.data_files
            for pub in pubs
        ]

    def __len__(self):
        return len(self.data_files)

    def __getitem__(self, idx):
        data_pub = self.data_files[idx]

        input_ids, attention_masks = get_sample_from_publication(
            data_pub,
            self.tokenizer,
            self.l_seq,
        )

        input_ids = torch.tensor(input_ids)
        attention_masks = torch.tensor(attention_masks)

        return dict(
            input_ids=input_ids,
            labels=input_ids,
            attention_mask=attention_masks,
        )
