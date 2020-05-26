import numpy as np
import torch
from torch.utils.data import TensorDataset, DataLoader
from dmelodies_dataset import DMelodiesDataset


class DMelodiesTorchDataset:
    def __init__(self, seed=0):
        self.kwargs = {'pin_memory': True} if torch.cuda.is_available() else {}
        self.dataset = None
        self.seed = seed
        self.note2index_dict = None
        self.index2note_dict = None
        self.latent_dicts = None
        np.random.seed(seed)

    def load_dataset(self):
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

    def data_loaders(self, batch_size, split=(0.70, 0.20)):
        """
        Returns three data loaders obtained by splitting
        self.dataset according to split
        :param batch_size:
        :param split:
        :return:
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
