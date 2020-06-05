"""
Script to create the dMelodies dataset
Running this script creates the following:
    - a .npz file containing the dataset and the indices of the latent values as numpy nd.arrays
            also contains dictionaries mapping note names to indices
    - a .csv file containing the latent factor information for all data points
Additional Options:
    Add the following arguments while running the script to save the generated melodies
    - '--save-mid': saves all melodies in .mid format
    - '--save-xml': saves all melodies in .musicxml format
    - '--debug': creates a smaller version of the dataset for debugging
See constants_file_names for information regarding file names and where they will be saved
"""

import os
import argparse
from joblib import Parallel, delayed
import pandas as pd
import multiprocessing
from tqdm import tqdm

from constants_file_names import RAW_DATA_FOLDER
from dmelodies_dataset import DMelodiesDataset
from helpers import get_score_for_item, get_file_name_for_item


def save_(
        index: int,
        data_row: pd.Series,
        save_mid: bool = False,
        save_xml: bool = False
):
    """
    Saves the score for the index as .mid and / or .musicxml
    Args:
        index: int, of the row
        data_row: single row of a pandas data-frame object containing the attribute values
        save_mid: bool, save as .mid if True
        save_xml: bool, save as .musicxml if True

    """
    if not (save_mid or save_xml):
        return
    score = get_score_for_item(data_row)
    file_name = get_file_name_for_item(data_row, index)
    if save_mid:
        midi_save_path = os.path.join(RAW_DATA_FOLDER, 'midi', file_name + '.mid')
        if not os.path.exists(os.path.dirname(midi_save_path)):
            os.makedirs(os.path.dirname(midi_save_path))
        score.write('midi', midi_save_path)
    if save_xml:
        xml_save_path = os.path.join(RAW_DATA_FOLDER, 'musicxml', file_name + '.musicxml')
        if not os.path.exists(os.path.dirname(xml_save_path)):
            os.makedirs(os.path.dirname(xml_save_path))
        score.write('musicxml', xml_save_path)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--save-mid', help='save data points in .mid format (default: false', action='store_true'
    )
    parser.add_argument(
        '--save-xml', help='save data points in .mid format (default: false', action='store_true'
    )
    parser.add_argument(
        '--debug', help='flag to create a smaller subset for debugging', action='store_true'
    )
    args = parser.parse_args()
    s_mid = args.save_mid
    s_xml = args.save_xml
    debug = args.debug

    # create and load dataset
    num_data_points=None
    if debug:
        num_data_points = 1000
    dataset = DMelodiesDataset(num_data_points=num_data_points)
    dataset.make_or_load_dataset()

    # save raw data-files if needed
    df = dataset.df.head(n=dataset.num_data_points)
    if debug:
        for i, d in tqdm(df.iterrows()):
            save_(i, d, s_mid, s_xml)
    else:
        cpu_count = multiprocessing.cpu_count()
        print(cpu_count)
        Parallel(n_jobs=cpu_count)(
            delayed(save_)(i, d, s_mid, s_xml) for i, d in tqdm(df.iterrows())
        )
