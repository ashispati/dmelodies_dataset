"""
Module containing the dMelodies torch dataloader
"""

import numpy as np
import torch
from torch.utils.data import TensorDataset, DataLoader
from typing import Tuple
from helpers import *
from dmelodies_dataset import DMelodiesDataset
from constants_latent_factors import *


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
        self.tick_durations = None
        self.beat_subdivisions = None
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
        self.beat_subdivisions = dataset.beat_subdivisions
        self.tick_durations = dataset.tick_durations

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

    def tensor_to_m21score(self, tensor_score):
        """
        Converts lead given as tensor_lead to a true music21 score
        :param tensor_score:
        :return:
        """
        slur_index = self.note2index_dict[SLUR_SYMBOL]

        score = music21.stream.Score()
        part = music21.stream.Part()
        # LEAD
        dur = 0
        f = music21.note.Rest()
        tensor_lead_np = tensor_score.cpu().numpy().flatten()
        tensor_lead_np[tensor_lead_np >= 52] = slur_index
        a = 1
        for tick_index, note_index in enumerate(tensor_lead_np):
            # if it is a played note
            if not note_index == slur_index:
                # add previous note
                if dur > 0:
                    f.duration = music21.duration.Duration(dur)
                    part.append(f)

                dur = self.tick_durations[tick_index % self.beat_subdivisions]
                f = standard_note(self.index2note_dict[note_index])
            else:
                dur += self.tick_durations[tick_index % self.beat_subdivisions]
        # add last note
        f.duration = music21.duration.Duration(dur)
        part.append(f)
        score.insert(part)
        return score

    @staticmethod
    def fix_note_str(note_str):
        note_map = {
            'B': 'A',
            'A': 'G',
            'E': 'D',
            'D': 'C'
        }
        new_note_str = ''
        if note_str[1] == '-':
            new_note_str += note_map[note_str[0]]
            new_note_str += '#'
            new_note_str += note_str[2]
        else:
            new_note_str = note_str
        return new_note_str

    def get_root_note(self, tensor_score):
        """
        Returns the root note for the given input score
        :param tensor_score: pytorch tensor, (16,)
        :return: music21.note.Note object
        """
        midi_pitch_array,  note_list = self.compute_midi_sequence(tensor_score)
        min_idx = np.argmin(midi_pitch_array[:3])
        root_note = note_list[min_idx]
        return root_note

    def compute_midi_sequence(self, tensor_score):
        """
        Returns a numpy array of midi pitch numbers given an input score
        :param tensor_score: tensor_score: pytorch tensor, (16,)
        :return: np.array, (L,)
        """
        # create MIDI pitch sequence
        slur_index = self.note2index_dict[SLUR_SYMBOL]
        rest_index = self.note2index_dict['rest']
        numpy_score = tensor_score.numpy()
        numpy_score[numpy_score >= 52] = slur_index
        numpy_score = numpy_score[numpy_score != rest_index]
        numpy_score = numpy_score[numpy_score != slur_index]
        midi_pitch_array = np.zeros_like(numpy_score)
        note_list = []
        for i in range(numpy_score.size):
            note_str = self.fix_note_str(self.index2note_dict[numpy_score[i]])
            n = music21.note.Note(note_str)
            note_list.append(n)
            midi_pitch_array[i] = n.pitch.midi

        return midi_pitch_array, note_list

    def compute_tonic_octave(self, tensor_score):
        """
        Computes the indices fo the tonic, octave for a given input score
        :param tensor_score: pytorch tensor, (16,)
        :return: tuple[int, int]
        """
        root_note = self.get_root_note(tensor_score)
        octave = OCTAVE_REVERSE_DICT[root_note.octave] if root_note.octave in OCTAVE_REVERSE_DICT.keys() else -1
        tonic = TONIC_REVERSE_DICT[root_note.name] if root_note.name in TONIC_REVERSE_DICT.keys() else -1
        return tonic, octave

    def compute_mode(self, tensor_score):
        """
        Computes the most likely mode for a given input score
        :param tensor_score:  pytorch tensor, (16,)
        :return: int
        """
        # get midi for root note
        root_midi = self.get_root_note(tensor_score).pitch.midi
        # get midi pitch sequence
        midi_pitch_array, _ = self.compute_midi_sequence(tensor_score)
        # create diff array
        diff_array = (midi_pitch_array - root_midi) % 12
        # compare diff array
        mode_idx = -1
        for mode in SCALE_NOTES_DICT.keys():
            scale_note_set = set(SCALE_NOTES_DICT[mode])
            if set(diff_array).issubset(scale_note_set):
                mode_idx = SCALE_REVERSE_DICT[mode]
                break
        return mode_idx

    def compute_rhythm(self, tensor_score, bar_num):
        """
        Computes the index for the rhythm for a given input score and bar number
        :param tensor_score: pytorch tensor, (16,)
        :param bar_num: int, 1 or 2
        :return: int
        """
        slur_index = self.note2index_dict[SLUR_SYMBOL]
        rest_index = self.note2index_dict['rest']
        if bar_num == 1:
            bar1_tensor = tensor_score[:8].clone().numpy()
        elif bar_num == 2:
            bar1_tensor = tensor_score[8:].clone().numpy()
        else:
            raise ValueError("Invalid bar number")
        bar1_tensor[bar1_tensor >= 52] = rest_index
        bar1_tensor[bar1_tensor == slur_index] = rest_index
        bar1_tensor[bar1_tensor == rest_index] = -1
        bar1_tensor[bar1_tensor != -1] = 1
        bar1_tensor[bar1_tensor == -1] = 0
        for rhy_idx in RHYTHM_DICT.keys():
            if list(bar1_tensor) == RHYTHM_DICT[rhy_idx]:
                return rhy_idx
        return -1

    def compute_arp_dir(self, tensor_score):
        """
        Computes the arpeggiation direction for a given input score
        :param tensor_score: pytorch tensor, (16,)
        :return: tuple[int, int, int, int]
        """
        midi_pitch_array, _ = self.compute_midi_sequence(tensor_score)
        arp_dir = [-1, -1, -1, -1]
        if midi_pitch_array.size == 12:
            midi_pitch_array = np.reshape(midi_pitch_array, (4, 3))
            diff_array = np.sign(np.diff(midi_pitch_array, axis=1))
            s_array = np.sum(diff_array, axis=1)
            for i in range(s_array.size):
                if s_array[i] > 0:
                    arp_dir[i] = ARP_REVERSE_DICT['up']
                elif s_array[i] < 0:
                    arp_dir[i] = ARP_REVERSE_DICT['down']
        return tuple(arp_dir)



