"""
Module containing the dMelodies torch dataloader
"""

import numpy as np
import torch
from torch.utils.data import TensorDataset, DataLoader
from typing import Tuple
from dmelodies_dataset import DMelodiesDataset


class DMelodiesTorchDataset:
    """
    Class defining a torch dataloader for the dMelodies dataset
    """
    def __init__(self, seed: int = 0):
        """
        Initializes the DMelodiesTorchDataset class
        Args:
            seed: int, specifies the random seed to be used for shuffling the data
        """
        self.kwargs = {'pin_memory': True} if torch.cuda.is_available() else {}
        self.dataset = None
        self.seed = seed
        self.note2index_dict = None
        self.index2note_dict = None
        self.latent_dicts = None
        np.random.seed(seed)

    def load_dataset(self):
        """
        Loads and shuffles the data
        """
        dataset = DMelodiesDataset()
        dataset.make_or_load_dataset()
        score = dataset.score_array
        score = np.expand_dims(score, axis=1)
        latent_values = dataset.latent_array
        a = np.c_[
            score.reshape(len(score), -1),
            latent_values.reshape(len(latent_values), -1)
        ]
        score2 = a[:, :score.size // len(score)].reshape(score.shape)
        latent_values2 = a[:, score.size // len(score):].reshape(latent_values.shape)
        np.random.shuffle(a)
        self.dataset = TensorDataset(
            torch.from_numpy(score2),
            torch.from_numpy(latent_values2)
        )
        self.note2index_dict = dataset.note2index_dict
        self.index2note_dict = dataset.index2note_dict
        self.latent_dicts = dataset.latent_dicts

    def data_loaders(
            self, batch_size: int, split: tuple = (0.70, 0.20)
    ) -> Tuple[DataLoader, DataLoader, DataLoader]:
        """
        Returns three data loaders obtained by splitting the data
        Args:
            batch_size: int, number of data points in each batch
            split: tuple, specify the ratio in which the dataset is to be divided
        Returns:
            tuple of 3 DataLoader objects corresponding to the train, validation and test sets
        """
        assert sum(split) < 1

        if self.dataset is None:
            self.load_dataset()

        num_examples = len(self.dataset)
        a, b = split
        train_dataset = TensorDataset(
            *self.dataset[: int(a * num_examples)]
        )
        val_dataset = TensorDataset(
            *self.dataset[int(a * num_examples):int((a + b) * num_examples)]
        )
        eval_dataset = TensorDataset(
            *self.dataset[int((a + b) * num_examples):]
        )

        train_dl = DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True,
            **self.kwargs
        )

        val_dl = DataLoader(
            val_dataset,
            batch_size=batch_size,
            shuffle=True,
        )

        eval_dl = DataLoader(
            eval_dataset,
            batch_size=batch_size,
            shuffle=False,
        )
        return train_dl, val_dl, eval_dl
