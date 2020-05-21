"""
File containing constants for different latent factors
Do NOT change
"""

import numpy as np
from itertools import combinations

# Dictionary for note indices corresponding to different chords within a scale
CHORD_DICT = {
    'I': [0, 2, 4],
    'IV': [3, 5, 7],
    'V': [4, 6, 8],
    'I-last': [0, 2, 4],
}

# List of allowed options for Tonic factor
TONIC_LIST = [
    'C',  'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B'
]
TONIC_DICT = {
    i: t for i, t in enumerate(TONIC_LIST)
}  # dict with indices mapping to the tonic values
TONIC_REVERSE_DICT = {
    TONIC_DICT[k]: k for k in TONIC_DICT.keys()
}  # reverse dict with tonic values mapping to indices

# List of allowed options for Octave factor
OCTAVE_LIST = [4, 5, 6]
OCTAVE_DICT = {
    i: o for i, o in enumerate(OCTAVE_LIST)
}  # dict with indices mapping to the octave values
OCTAVE_REVERSE_DICT = {
    OCTAVE_DICT[k]: k for k in OCTAVE_DICT.keys()
}  # reverse dict with octave values mapping to indices

# List of allowed options for Scale (Mode) factor
SCALE_LIST = ['major', 'minor', 'blues']
SCALE_DICT = {
    i: m for i, m in enumerate(SCALE_LIST)
}  # dict with indices mapping to the scale options
SCALE_REVERSE_DICT = {
    SCALE_DICT[k]: k for k in SCALE_DICT.keys()
}  # reverse dict with scale values mapping to indices
SCALE_NOTES_DICT = {
    'major': [0, 2, 4, 5, 7, 9, 11, 12, 14],
    'minor': [0, 2, 3, 5, 7, 8, 11, 12, 14],
    'blues': [0, 2, 3, 5, 6, 9, 10, 12, 14],
}  # dict with allowed scale degrees for each scale

# Dict containing options for Rhythm factor
RHYTHM_DICT = {}
all_rhythms = combinations([0, 1, 2, 3, 4, 5, 6, 7], 6)
for i, pos in enumerate(list(all_rhythms)):
    temp_array = np.array([0] * 8)
    temp_array[np.array(pos)] = 1
    RHYTHM_DICT[i] = list(temp_array)

# Dict containing options for Arpeggiation factor
ARP_DICT = {
    0: 'up',
    1: 'down'
}  # dict with indices mapping to arpeggiation direction options
ARP_REVERSE_DICT = {
    ARP_DICT[k]: k for k in ARP_DICT.keys()
}  # reverse dict mapping arpeggiaition direction options to indices
