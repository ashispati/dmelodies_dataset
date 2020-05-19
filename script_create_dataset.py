"""
Script to create the dMelodies dataset
Running this script creates the following:
    - a .json file containing dicts for mapping indices to musical note names
    - a .npz file containing the dataset and the indices of the latent values as numpy nd.arrays
    - a .csv file containing the latent factor information for all data points
Additional Options (to be included):
    - '': saves all melodies in .musicxml format
    - '': saves all melodies in .mid format
See constants_file_names for information regarding file names and where they will be saved
"""

import os
from itertools import product
from joblib import Parallel, delayed
import pandas as pd
from tqdm import tqdm

from constants_latent_factors import *
from constants_file_names import *


def create_latent_info_df():
    """
    Creates and returns the dataframe object containing the latent factors information

    Returns:
         pandas dataframe object
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
    for i, params in tqdm(enumerate(all_combinations)):
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


def get_latent_info():
    """
    Reads the latent factors info from stored LATENT_INFO_CSV (see constants_file_names.py) file.
    If file doesn't exist, creates and saves it

    Returns:
        pandas dataframe object
    """
    if os.path.exists(LATENT_INFO_CSV):
        latent_df = pd.read_csv('dmelodies_dataset_latent_info.csv', index_col=0)
    else:
        latent_df = create_latent_info_df()
        latent_df.to_csv(path_or_buf=LATENT_INFO_CSV)
    return latent_df


if __name__ == '__main__':
    df = get_latent_info()
