[![License: CC BY-NC-SA 4.0](https://img.shields.io/badge/License-CC%20BY--NC--SA%204.0-ff69b4.svg)](http://creativecommons.org/licenses/by-nc-sa/4.0/)

# dMelodies - Disentanglement testing Melodies dataset

## Motivation
Over the last few years, there has been significant research attention on representation learning focused on disentangling the underlying factors of variation in given data. However, most of the current/previous studies rely on datasets from the image/computer vision domain (such as the [dSprites](https://github.com/deepmind/dsprites-dataset) dataset).
The purpose of this work is to be able to create a standardized dataset for conducting disentanglement studies on symbolic music data. The key motivation is that such a dataset would help researchers working on disentanglement problems demonstrate their algorithm on diverse domains. 

## Description
dMelodies is dataset of simple 2-bar melodies generated using 9 independent latent factors of variation where each data point represents a unique melody based on the following constraints:
- Each melody will correspond to a unique scale (major, minor, blues, etc.).
- Each melody plays the arpeggios using the standard I-V-IV-I cadence chord pattern.
- Bar 1 plays the first 2 chords (6 notes), Bar 2 plays the second 2 chords (6 notes).
- Each played note is an 8th note. 
 
A typical example is shown above. 
<p align="center">
    <img src=figs/dataset_example.svg width=500px alt="Example melody from the dataset"><br>
</p>


## Factors of Variation
The following factors of variation are considered: 
1. **Tonic** (Root): 12 options from C to B 
2. **Octave**: 3 options from C4 through C6
3. **Mode/Scale**: 3 options (Major, Minor, Blues) 
4. **Rhythm Bar 1**: 28 options based on where the 6 note onsets are located in the first bar.
5. **Rhythm Bar 2**: 28 options based on where the 6 note onsets are located in the second bar. 
6. **Arpeggiation Direction Chord 1**: 2 options (up/down) based on how the arpreggio is played
7. **Arpeggiation Direction Chord 2**: 2 options (up/down) 
8. **Arpeggiation Direction Chord 3**: 2 options (up/down)
9. **Arpeggiation Direction Chord 4**: 2 options (up/down)

Consequently, the total number of data-points are 1,354,752.

## Usage
Install `anaconda` or `miniconda` by following the instruction [here](https://docs.conda.io/projects/conda/en/latest/user-guide/install/).

Create a new conda environment using the `enviroment.yml` file located in the root folder of this repository. The instructions for the same can be found [here](https://docs.conda.io/projects/conda/en/latest/user-guide/tasks/manage-environments.html#creating-an-environment-from-an-environment-yml-file).

Activate the `dmelodies` environment using the following command:

```
conda activate dmelodies
```

To install, either download / clone this repository. Open a new terminal, `cd` into the root folder of this repository and run the following command

    pip install -e .

For creating the dataset (as a `.npz` file), run `script_create_dataset.py` from the root folder of this repository. This will create a `dataset` folder for saving the dataset. Additional arguments `--save-midi` and `--save-xml` can be used to save the individual melodies as `.mid` or `.musicxml` files. The files will be saved in a `raw_data` folder. 

The `DMelodiesDataset` class (see `dmelodies_dataset.py`) is a wrapper around the dataset and provides methods to read and load the dataset. If you are using [PyTorch](https://pytorch.org), you are use the `DMelodiesTorchDataset` class (see `dmelodies_torch_dataloader.py`) which implements a torch DataLoader.

## Attribution
TO BE UPDATED
