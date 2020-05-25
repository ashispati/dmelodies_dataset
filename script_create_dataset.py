"""
Script to create the dMelodies dataset
Running this script creates the following:
    - a .npz file containing the dataset and the indices of the latent values as numpy nd.arrays
            also contains dictionaries mapping note names to indices
    - a .csv file containing the latent factor information for all data points
Additional Options (to be included):
    - '': saves all melodies in .musicxml format
    - '': saves all melodies in .mid format
See constants_file_names for information regarding file names and where they will be saved
"""
from dmelodies_dataset import DMelodiesDataset


if __name__ == '__main__':
    dataset = DMelodiesDataset(num_data_points=10)
    dataset.make_or_load_dataset()
