"""
Module containing the DMelodiesDataset class
"""
import os
import json
import pandas as pd
from typing import Union

from helpers import *


class DMelodiesDataset:
    """
    Class for creating the dMelodies dataset
    """
    def __init__(self, num_data_points=None):
        """
        Initializes the DMelodiesDataset class object
        Args:
            num_data_points: int or None, if None create the full dataset,
        """
        self.df = get_latent_info()
        if num_data_points is None:
            self.num_data_points = self.df.shape[0]
        else:
            self.num_data_points = num_data_points
        self.time_sig_num = 4
        self.time_sig_den = 4
        self.beat_subdivisions = len(TICK_VALUES)
        self.tick_durations = compute_tick_durations(TICK_VALUES)
        self.dataset_path = os.path.join(DATASETS_FOLDER, NPZ_DATASET)
        if not os.path.exists(DATASETS_FOLDER):
            os.mkdir(DATASETS_FOLDER)
        self.score_array = None
        self.latent_array = None
        self.latent_dicts = {
            'tonic': TONIC_DICT,
            'octave': OCTAVE_DICT,
            'scale': SCALE_DICT,
            'rhythm': RHYTHM_DICT,
            'arp_dir': ARP_DICT
        }
        self.note2index_dict = dict()
        self.index2note_dict = dict()
        self.initialize_index_dicts()

    def _get_score_for_item(self, index) -> music21.stream.Score:
        """
        Returns the score for the index
        Args:
            index: int, of the item in the dataset

        Returns:
            music21.stream.Score object
        """
        assert 0 <= index < self.num_data_points
        d = self.df.iloc[index]
        return get_score_for_item(d)

    def _get_file_name_for_item(self, index) -> str:
        """
        Return the file name for index
        Args:
            index: int, of the item in the dataset

        Returns:
            str,
        """
        assert 0 <= index < self.num_data_points
        d = self.df.iloc[index]
        return get_file_name_for_item(d, index)

    def make_or_load_dataset(self):
        """
        Creates the dataset or reads if it already exists
        Returns:
            None
        """
        # read dataset if already exists
        if os.path.exists(self.dataset_path):
            print('Dataset already created. Reading it now')
            dataset = np.load(self.dataset_path, allow_pickle=True)
            self.score_array = dataset['score']
            self.latent_array = dataset['latent_values']
            self.note2index_dict = dataset['note2index_dict'].item()
            self.index2note_dict = dataset['index2note_dict'].item()
            self.latent_dicts = dataset['latent_dicts'].item()
            return

        # else, create dataset
        print('Making tensor dataset')
        score_seq = [None] * self.num_data_points
        latent_seq = [None] * self.num_data_points

        def _create_data_point(item_index):
            m21_score = self._get_score_for_item(item_index)
            score_array = self.get_tensor(m21_score)
            score_seq[item_index] = score_array
            latent_array = self._get_latents_array_for_index(item_index)
            latent_seq[item_index] = latent_array

        for idx in tqdm(range(self.num_data_points)):
            _create_data_point(idx)

        self.score_array = np.array(score_seq)
        self.latent_array = np.array(latent_seq)
        print('Number of data points: ', self.score_array.shape[0])
        np.savez(
            self.dataset_path,
            score=score_seq,
            latent_values=latent_seq,
            note2index_dict=self.note2index_dict,
            index2note_dict=self.index2note_dict,
            latent_dicts=self.latent_dicts,
        )

    def get_tensor(self, score) -> Union[np.array, None]:
        """
        Returns the score as a torch tensor

        Args:
            score: music21.stream.Score object

        Returns:
            torch.Tensor
        """
        eps = 1e-5
        notes = get_notes(score)
        if not is_score_on_ticks(score, TICK_VALUES):
            return None
        list_note_strings_and_pitches = [(n.nameWithOctave, n.pitch.midi)
                                         for n in notes
                                         if n.isNote]
        for note_name, pitch in list_note_strings_and_pitches:

            if note_name not in self.note2index_dict:
                self.update_index_dicts(note_name)

        # construct sequence
        x = 0
        y = 0
        length = int(score.highestTime * self.beat_subdivisions)
        t = np.zeros((length, 2))
        is_articulated = True
        num_notes = len(notes)
        current_tick = 0
        while y < length:
            if x < num_notes - 1:
                if notes[x + 1].offset > current_tick + eps:
                    t[y, :] = [self.note2index_dict[standard_name(notes[x])],
                               is_articulated]
                    y += 1
                    current_tick += self.tick_durations[
                        (y - 1) % len(TICK_VALUES)]
                    is_articulated = False
                else:
                    x += 1
                    is_articulated = True
            else:
                t[y, :] = [self.note2index_dict[standard_name(notes[x])],
                           is_articulated]
                y += 1
                is_articulated = False
        lead = t[:, 0] * t[:, 1] + (1 - t[:, 1]) * self.note2index_dict[SLUR_SYMBOL]
        lead = lead.astype('int32')
        return lead

    def _get_latents_array_for_index(self, index) -> np.array:
        """
        Returns the latent arrays from the file name
        Args:
            index: int,
        Returns:
            np.array containing the latent values
        """
        assert 0 <= index < self.num_data_points
        d = self.df.iloc[index]
        latent_list = [
            TONIC_REVERSE_DICT[d['tonic']],
            OCTAVE_REVERSE_DICT[d['octave']],
            SCALE_REVERSE_DICT[d['scale']],
            d['rhythm_bar1'],
            d['rhythm_bar2'],
            ARP_REVERSE_DICT[d['arp_chord1']],
            ARP_REVERSE_DICT[d['arp_chord2']],
            ARP_REVERSE_DICT[d['arp_chord3']],
            ARP_REVERSE_DICT[d['arp_chord4']],
        ]
        return np.array(latent_list).astype('int32')

    def initialize_index_dicts(self):
        """
        Reads index dicts from file if available, else creates it

        """
        note_sets = set()
        # add rest and slur symbols
        note_sets.add(SLUR_SYMBOL)
        note_sets.add('rest')
        for note_index, note_name in enumerate(note_sets):
            self.index2note_dict.update({note_index: note_name})
            self.note2index_dict.update({note_name: note_index})

    def update_index_dicts(self, new_note_name):
        """
        Updates self.note2index_dicts and self.index2note_dicts

        """
        new_index = len(self.note2index_dict)
        self.index2note_dict.update({new_index: new_note_name})
        self.note2index_dict.update({new_note_name: new_index})
        print(
            f'Warning: Entry {str({new_index: new_note_name})} added to dictionaries'
        )
