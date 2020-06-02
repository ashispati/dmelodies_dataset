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
See constants_file_names for information regarding file names and where they will be saved
"""

import argparse
from joblib import Parallel, delayed
import multiprocessing
from tqdm import tqdm

from dmelodies_dataset import DMelodiesDataset


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--save-mid', help='save data points in .mid format (default: false', action='store_true'
    )
    parser.add_argument(
        '--save-xml', help='save data points in .mid format (default: false', action='store_true'
    )
    args = parser.parse_args()
    save_mid = args.save_mid
    save_xml = args.save_xml

    dataset = DMelodiesDataset(num_data_points=10)
    dataset.make_or_load_dataset()

    cpu_count = multiprocessing.cpu_count()
    Parallel(n_jobs=cpu_count)(
        delayed(dataset.save)(i) for i in tqdm(range(dataset.num_data_points))
    )
