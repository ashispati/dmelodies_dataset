"""
Helper functions to create melodies and music21 score objects
"""

import os
from fractions import Fraction
from itertools import product
import pandas as pd
from tqdm import tqdm
from typing import Union

import music21
from music21 import note

from constants_file_names import *
from constants_latent_factors import *


SLUR_SYMBOL = '__'
TICK_VALUES = [
    0,
    Fraction(1, 2),
]


def create_latent_info_df() -> pd.DataFrame:
    """
    Creates and returns the data-frame object containing the latent factors information

    Returns:
         pandas data-frame object
    """
    tonic_list = []
    octave_list = []
    scale_list = []
    rhy1_list = []
    rhy2_list = []
    dir1_list = []
    dir2_list = []
    dir3_list = []
    dir4_list = []

    all_combinations = product(
        TONIC_DICT.keys(),
        OCTAVE_DICT.keys(),
        SCALE_DICT.keys(),
        RHYTHM_DICT.keys(),
        RHYTHM_DICT.keys(),
        ARP_DICT.keys(),
        ARP_DICT.keys(),
        ARP_DICT.keys(),
        ARP_DICT.keys()
    )
    for params in tqdm(all_combinations):
        tonic_list.append(TONIC_DICT[params[0]])
        octave_list.append(OCTAVE_DICT[params[1]])
        scale_list.append(SCALE_DICT[params[2]])
        rhy1_list.append(params[3])
        rhy2_list.append(params[4])
        dir1_list.append(ARP_DICT[params[5]])
        dir2_list.append(ARP_DICT[params[6]])
        dir3_list.append(ARP_DICT[params[7]])
        dir4_list.append(ARP_DICT[params[8]])
    d = {
        'tonic': tonic_list,
        'octave': octave_list,
        'scale': scale_list,
        'rhythm_bar1': rhy1_list,
        'rhythm_bar2': rhy2_list,
        'arp_chord1': dir1_list,
        'arp_chord2': dir2_list,
        'arp_chord3': dir3_list,
        'arp_chord4': dir4_list
    }
    latent_df = pd.DataFrame(data=d)
    return latent_df


def get_latent_info() -> pd.DataFrame:
    """
    Reads the latent factors info from stored LATENT_INFO_CSV (see constants_file_names.py) file.
    If file doesn't exist, creates and saves it

    Returns:
        pandas data-frame object
    """
    cur_dir = os.path.dirname(os.path.realpath(__file__))
    latent_info_path = os.path.join(cur_dir, LATENT_INFO_CSV)
    if os.path.exists(latent_info_path):
        latent_df = pd.read_csv(latent_info_path, index_col=0)
    else:
        latent_df = create_latent_info_df()
        latent_df.to_csv(path_or_buf=latent_info_path)
    return latent_df


def get_midi_pitch_list(
        tonic: str,
        octave: int,
        mode: str,
        arp_dir1: str,
        arp_dir2: str,
        arp_dir3: str,
        arp_dir4: str
) -> list:
    """
    Create the sequence of midi pitch values. Refer constants_latent_factors.py for details regarding allowed arg values
    Args:
        tonic: str, specifies the pitch class of the root note (C, C#, ...., through B)
        octave: int, specifies of the octave number (4, 5, 6) of the root note
        mode: str, specifies the scale (major, minor, blues etc.)
        arp_dir1: str, 'up' or 'down', specifies the arpeggiation direction of Chord 1
        arp_dir2: str, 'up' or 'down', specifies the arpeggiation direction of Chord 2
        arp_dir3: str, 'up' or 'down', specifies the arpeggiation direction of Chord 3
        arp_dir4: str, 'up' or 'down', specifies the arpeggiation direction of Chord 4
    Returns:
        list of MIDI notes corresponding to the melody defined based on the input arguments
    """
    root_pitch = music21.pitch.Pitch(tonic + str(octave)).midi
    pitch_seq = []
    dir_seq = [arp_dir1, arp_dir2, arp_dir3, arp_dir4]
    for index, chord in enumerate(CHORD_DICT.keys()):
        seq = CHORD_DICT[chord]
        if dir_seq[index] == 'down':
            seq = seq[::-1]
        for s in seq:
            midi_pitch = root_pitch + SCALE_NOTES_DICT[mode][s]
            pitch_seq.append(midi_pitch)
    return pitch_seq


def create_m21_melody(
        tonic: str,
        octave: int,
        mode: str,
        rhythm_bar1: int,
        rhythm_bar2: int,
        arp_dir1: str,
        arp_dir2: str,
        arp_dir3: str,
        arp_dir4: str
) -> music21.stream.Score:
    """
    Creates the 2-bar melody in music21 score format
    Args:
        tonic: str, specifies the pitch class of the root note (C, C#, ...., through B)
        octave: int, specifies of the octave number (4, 5, 6) of the root note
        mode: str, specifies the scale (major, minor, blues etc.)
        rhythm_bar1: int, specified the rhythm for Bar 1
        rhythm_bar2: int, specified the rhythm for Bar 2
        arp_dir1: str, 'up' or 'down', specifies the arpergiation direction of Chord 1
        arp_dir2: str, 'up' or 'down', specifies the arpeggiation direction of Chord 2
        arp_dir3: str, 'up' or 'down', specifies the arpeggiation direction of Chord 3
        arp_dir4: str, 'up' or 'down', specifies the arpeggiation direction of Chord 4

    Returns:
        music21 score object containing the score
    """
    score = music21.stream.Score()
    part = music21.stream.Part()
    dur = 0.0
    rhy1 = RHYTHM_DICT[rhythm_bar1]
    rhy2 = RHYTHM_DICT[rhythm_bar2]
    if sum(rhy1) != 6:
        raise(ValueError, f'Invalid rhythm: {rhy1}')
    if sum(rhy2) != 6:
        raise(ValueError, f'Invalid rhythm: {rhy2}')
    midi_pitch_seq = get_midi_pitch_list(tonic, octave, mode, arp_dir1, arp_dir2, arp_dir3, arp_dir4)
    curr_note_num = 0
    for rhy in [rhy1, rhy2]:
        for onset in rhy:
            if onset == 1:
                f = music21.note.Note()
                f.pitch.midi = midi_pitch_seq[curr_note_num]
                f.duration = music21.duration.Duration('eighth')
                curr_note_num += 1
            else:
                f = music21.note.Rest()
                f.duration = music21.duration.Duration('eighth')
            part.insert(dur, f)
            dur += music21.duration.Duration('eighth').quarterLength

    score.insert(part)
    return score


def get_score_for_item(df_row: pd.Series) -> music21.stream.Score:
    """
    Returns the score for the index given a data-frame
    Args:
        df_row: data-frame row containing latent attribute values

    Returns:
        music21.stream.Score object
    """
    return create_m21_melody(
        tonic=df_row['tonic'],
        octave=df_row['octave'],
        mode=df_row['scale'],
        rhythm_bar1=df_row['rhythm_bar1'],
        rhythm_bar2=df_row['rhythm_bar2'],
        arp_dir1=df_row['arp_chord1'],
        arp_dir2=df_row['arp_chord2'],
        arp_dir3=df_row['arp_chord3'],
        arp_dir4=df_row['arp_chord4']
    )


def get_file_name_for_item(df_row: pd.Series, index: int) -> str:
    """
    Return the file name for index
    Args:
        df_row: data-frame row containing latent attribute values
        index: int, of the item in the dataset

    Returns:
        str,
    """
    tonic = df_row['tonic']
    octave = df_row['octave']
    mode = df_row['scale']
    rhythm_bar1 = df_row['rhythm_bar1']
    rhythm_bar2 = df_row['rhythm_bar2']
    dir1 = df_row['arp_chord1']
    dir2 = df_row['arp_chord2']
    dir3 = df_row['arp_chord3']
    dir4 = df_row['arp_chord4']
    file_name = f'{index}_{tonic}_{octave}_{mode}_{rhythm_bar1}_{rhythm_bar2}_{dir1}_{dir2}_{dir3}_{dir4}'
    return file_name


def compute_tick_durations(tick_values: list):
    """
    Computes the tick durations
    Args:
        tick_values: list of allowed tick values
    """
    diff = [n - p
            for n, p in zip(tick_values[1:], tick_values[:-1])]
    diff = diff + [1 - tick_values[-1]]
    return diff


def get_notes(score: music21.stream.Score) -> list:
    """
    Returns the notes from the music21 score object
    Args:
        score: music21 score object
    Returns:
        list, of music21 note objects
    """
    notes = score.parts[0].flat.notesAndRests
    notes = [n for n in notes if not isinstance(n, music21.harmony.ChordSymbol)]
    return notes


def is_score_on_ticks(score: music21.stream.Score, tick_values: list) -> bool:
    """
    Checks if the notes in a score are on ticks
    Args:
        score: music21 score object
        tick_values: list of allowed tick values
    """
    notes = get_notes(score)
    eps = 1e-5
    for n in notes:
        _, d = divmod(n.offset, 1)
        flag = False
        for tick_value in tick_values:
            if tick_value - eps < d < tick_value + eps:
                flag = True
        if not flag:
            return False
    return True


def standard_name(note_or_rest: Union[music21.note.Note, music21.note.Rest]) -> str:
    """
    Converts music21 note objects to string
    Args:
        note_or_rest: music21 note.Note or note.Rest object

    Returns:
        str,
    """
    if isinstance(note_or_rest, music21.note.Note):
        return note_or_rest.nameWithOctave
    elif isinstance(note_or_rest, music21.note.Rest):
        return note_or_rest.name
    else:
        raise ValueError("Invalid input. Should be a music21.note.Note or music21.note.Rest object ")


def standard_note(note_or_rest_string: str) -> Union[music21.note.Note, music21.note.Rest]:
    """
    Converts str to music21 note.Note or note.Rest object
    Args:
        note_or_rest_string:

    Returns:
        music21 note.Note or note.Rest object
    """
    if note_or_rest_string == 'rest':
        return note.Rest()
    elif note_or_rest_string == SLUR_SYMBOL:
        return note.Rest()
    else:
        return note.Note(note_or_rest_string)
