import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader
from collections import Counter
import re
import os


class Vocabulary:
    """
    vocabulary class for converting between tokens and indices.
    """

    def __init__(self, max_size=10000):
        self.max_size = max_size
        self.token2idx = {"<PAD>": 0, "<UNK>": 1, "<CLS>": 2, "<SEP>": 3}
        self.idx2token = {0: "<PAD>", 1: "<UNK>", 2: "<CLS>", 3: "<SEP>"}
        self.token_counts = Counter()

    def add_token(self, token):
        """add a token to the vocabulary"""
        self.token_counts[token] += 1

    def build_vocab(self):
        """build vocabulary from token counts"""
        # sort tokens by frequency (descending) and add to vocabulary
        most_common = self.token_counts.most_common(self.max_size - len(self.token2idx))
        for token, _ in most_common:
            idx = len(self.token2idx)
            self.token2idx[token] = idx
            self.idx2token[idx] = token

    def convert_tokens_to_ids(self, tokens):
        """convert a list of tokens to indices"""
        return [self.token2idx.get(token, self.token2idx["<UNK>"]) for token in tokens]

    def convert_ids_to_tokens(self, ids):
        """convert a list of indices to tokens"""
        return [self.idx2token.get(idx, "<UNK>") for idx in ids]

    def __len__(self):
        return len(self.token2idx)


class TextDataset(Dataset):
    """
    dataset for text classification tasks.
    """

    def __init__(
        self, texts, labels=None, vocab=None, max_seq_len=128, build_vocab=True
    ):
        self.texts = texts
        self.labels = labels
        self.max_seq_len = max_seq_len

        # tokenize texts
        self.tokenized_texts = [self.tokenize(text) for text in texts]

        # build or use provided vocabulary
        if vocab is None and build_vocab:
            self.vocab = Vocabulary()
            for tokens in self.tokenized_texts:
                for token in tokens:
                    self.vocab.add_token(token)
            self.vocab.build_vocab()
        else:
            self.vocab = vocab

        # convert tokens to indices
        self.encoded_texts = [self.encode(tokens) for tokens in self.tokenized_texts]

    def tokenize(self, text):
        """simple tokenization by splitting on whitespace and removing punctuation"""
        # convert to lowercase and remove punctuation
        text = re.sub(r"[^\w\s]", "", text.lower())
        # split on whitespace
        return text.split()

    def encode(self, tokens):
        """Encode a list of tokens as indices with CLS token and padding"""
        # add CLS token at the beginning
        ids = [self.vocab.token2idx["<CLS>"]]

        # convert tokens to ids and add (up to max_seq_len - 2)
        token_ids = self.vocab.convert_tokens_to_ids(tokens)
        ids.extend(token_ids[: self.max_seq_len - 2])

        # add SEP token at the end
        ids.append(self.vocab.token2idx["<SEP>"])

        # pad to max_seq_len
        padding_length = self.max_seq_len - len(ids)
        if padding_length > 0:
            ids.extend([self.vocab.token2idx["<PAD>"]] * padding_length)

        return ids

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        item = {"input_ids": torch.tensor(self.encoded_texts[idx], dtype=torch.long)}

        if self.labels is not None:
            item["label"] = torch.tensor(self.labels[idx], dtype=torch.long)

        return item


def create_data_loaders(
    train_texts, train_labels, test_texts, test_labels, batch_size=32, max_seq_len=128
):
    """
    create train and test data loaders for a text classification task.

    Args:
        train_texts: list of training text samples
        train_labels: list of training labels
        test_texts: list of test text samples
        test_labels: list of test labels
        batch_size: batch size for data loaders
        max_seq_len: max sequence length

    Returns:
        train_loader: DataLoader for training data
        test_loader: DataLoader for test data
        vocab: vocabulary object
    """
    # create training dataset and build vocabulary
    train_dataset = TextDataset(train_texts, train_labels, max_seq_len=max_seq_len)
    vocab = train_dataset.vocab

    # create test dataset with the same vocabulary
    test_dataset = TextDataset(
        test_texts, test_labels, vocab=vocab, max_seq_len=max_seq_len, build_vocab=False
    )

    # create data loaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size)

    return train_loader, test_loader, vocab


def load_sample_data():
    """
    load a small sample dataset for sentiment analysis.
    i'm too lazy to actually find and use a dataset lol

    Returns:
        train_texts: List of training text samples
        train_labels: List of training labels
        test_texts: List of test text samples
        test_labels: List of test labels
    """
    # sample positive and negative reviews
    positive_samples = [
        "This movie was fantastic! I really enjoyed it.",
        "Great performance by all the actors. Highly recommended.",
        "One of the best films I have seen in years.",
        "The plot was engaging and the characters were well developed.",
        "I loved everything about this film, from start to finish.",
    ]

    negative_samples = [
        "This was a terrible waste of time.",
        "I was so bored throughout the entire movie.",
        "The acting was poor and the story made no sense.",
        "Don't waste your money on this awful film.",
        "I've never seen anything so disappointing.",
    ]

    # create training set (8 samples)
    train_texts = positive_samples[:4] + negative_samples[:4]
    train_labels = [1] * 4 + [0] * 4  # 1 for positive, 0 for negative

    # create test set (2 samples)
    test_texts = positive_samples[4:] + negative_samples[4:]
    test_labels = [1] * 1 + [0] * 1

    return train_texts, train_labels, test_texts, test_labels
