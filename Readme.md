# Morphodynamics

This software can be used to analyze the dynamics of single-cells imaged by time-lapse fluorescence microscopy. The dynamics of morphology and fluorescence intensity distribution can be jointly analyzed through the segmentation and splitting of cells into a set of smaller regions (windows) that can be represented as two-dimensional grids to facilitate interpretation of events.

This software has been developed at Bern University by Cédric Vonesch (Science IT Support and Pertz lab) and Guillaume Witz (Microscopy Imaging Center and Science IT Support) with the collaboration of Jakobus van Unen (Pertz lab)**.

**For a complete documentation see the [Morphodynamics website](https://guiwitz.github.io/MorphoDynamics/mydocs/Introduction.html).

## Installation

We strongly recommend to install the necessary software via conda. If you don't have conda installed, follow [these instructions](https://docs.conda.io/en/latest/miniconda.html) to install a minimal version called miniconda.

Then, download (or clone) this repository to your machine. If you are working on a local laptopn you can use the green "Code" button at the top right of this page for download and then unzip the folder. If you are working on a remote machine you can type:

```
git clone https://github.com/guiwitz/MorphoDynamics.git
```

Open a terminal and move to the downloaded folder (Morphodynamics-master). The latter contains an ```environment.yml``` file that can be used to setup a conda environment wih all necessary packages. For that, just execute the following line:

```
conda env create -f environment.yml
```

Then activate the environment:

```
conda activate morphodynamics
```

The Morphodynamics package is automatically installed in that environment.


## Updates

To update your local installation with the latest version available on GitHub, activate the environment and install the package directly from GitHub:

```
conda activate morphodynamics 
pip install --upgrade git+https://github.com/guiwitz/MorphoDynamics.git@master#egg=morphodynamics
```

The above command should prompt you to enter your GitHub username and password, as the repository is private.

Note: close all notebooks (click on File | Close and Halt) prior to the update and reopen them afterwards.

## Usage

Whenever you want to use Jupyter and the Morphodynamics package, open a terminal, activate the environment 

```
conda activate morphodynamics
```

and start a Jupyter session:

```
jupyter notebook
```

Two notebooks are provided in the [notebooks](notebooks) folder. [Morpho_segmentation.ipynb](notebooks/Morpho_segmentation.ipynb) allows you to perform cell segmentation and windowing. It accepts data in the form of series of tiffs, tiff stacks or nd2 files (still experimental). Once segmentation is done and saved, that information can be used to proceed to the data analysis per se in the [InterfaceFigures.ipynb](notebooks/InterfaceFigures.ipynb) notebooks. There you import the segmentation, and can choose from a variety of different analysis to plot.

## Development

When releasing a new version v.X.Y.Z, bump the version in the [version.txt](morphodynamics/version.txt) file, commit-push, and create an annotated tag:

```
git tag -m "git versioning" -a X.Y.Z
git push origin X.Y.Z
```

When updating the master branch in between releases, automatically bump the version in the [version.txt](morphodynamics/version.txt) file by using the [version.py](morphodynamics/version.py) module: activate the environment, move to the project folder and execute:

```
python morphodynamics/version.py
```

The new version will be listed in ```conda list``` and the ```morphodynamics.__version__``` variable.