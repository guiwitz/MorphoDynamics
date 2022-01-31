# Morphodynamics <img src="/images/logo.png" alt="alt text" width="50">

[![License](https://img.shields.io/pypi/l/morphodynamics?color=green)](https://github.com/guiwitz/morphodynamics/raw/master/LICENSE)
[![PyPI](https://img.shields.io/pypi/v/morphodynamics.svg?color=green)](https://pypi.org/project/morphodynamics)
[![Python Version](https://img.shields.io/pypi/pyversions/morphodynamics.svg?color=green)](https://python.org)
[![tests](https://github.com/guiwitz/morphodynamics/workflows/tests/badge.svg)](https://github.com/guiwitz/morphodynamics/actions)


This software can be used to analyze the dynamics of single-cells imaged by time-lapse fluorescence microscopy. The dynamics of morphology and fluorescence intensity distribution can be jointly analyzed through the segmentation and splitting of cells into a set of smaller regions (windows) that can be represented as two-dimensional grids to facilitate interpretation of events.

This software has been developed at Bern University by Cédric Vonesch (Science IT Support and Pertz lab) and Guillaume Witz (Microscopy Imaging Center and Science IT Support) with the collaboration of Jakobus van Unen (Pertz lab).

**For a complete documentation see the [Morphodynamics website](https://guiwitz.github.io/MorphoDynamics/mydocs/Introduction.html)**.

## Install the package

You can simply install the package using

```
pip install morphodynamics
```
To use the napari interface, you will have in addition to install the ```napari-morphodynamics``` plugin:

```
pip install git+https://github.com/guiwitz/napari-morphodynamics.git
```

### Notes
#### Versions

The Morphodynamics package had undergone a massive change between version ```0.2.4``` and ```0.3.0```, in particular with a new interface in napari and a more customized way of generating post-processing plots. If you want to install the old 0.2.x series version, please use:

```
pip install git+https://github.com/guiwitz/morphodynamics.git@v0.2.4
```

#### nd2
The default ```nd2reader``` sometimes fails to read files containing non-rectanuglar rois. In such cases, you can try to install instead a customized version of the reader:

```
pip install git+https://github.com/guiwitz/nd2reader.git@master#egg=nd2reader
```

## Package usage

This package can be used entirely programmatically to process data via its API. Examples for this can be found in the docs [here](https://guiwitz.github.io/MorphoDynamics/mydocs/Analysis_without_UI.html) and [here](https://guiwitz.github.io/MorphoDynamics/mydocs/usage_step_by_step.html).
## napari-morphodynamics plugin

The napari-morphodynamics plugin offers an interface to import, visualize and process data. A detailed description for its usage can be found in the [docs](https://guiwitz.github.io/MorphoDynamics/mydocs/Napari_Plugin.html). To install the plugin you can use:

```
pip pip install git+https://github.com/guiwitz/napari-morphodynamics.git
```

## Notebooks

In addition, the analysis can be run in the interactive notebook [Morpho_segmentation.ipynb](https://guiwitz.github.io/MorphoDynamics/Morpho_segmentation.ipynb). This can be useful when e.g. working remotely on a cluster.

## Post-processing

Once the images have been processed the output results can be used to plot various features of the experiment such as cell shape evolution, edge displacement, per window signal evolution etc. Examples on how to create such plots are given in the [Postprocessing.ipynb](https://guiwitz.github.io/MorphoDynamics/Postprocessing.ipynb) notebook.

## Environment

To ensure that you have all recommended packages to perform interactive work, we recommend to create en environment with conda. If you don't have conda installed yet, follow [these instructions](https://docs.conda.io/en/latest/miniconda.html) to install a minimal version called miniconda.

To create the appropriate environment that will for example also contain the optional dependency ```cellpose``` for cell segmentation, save the following [environment.yml](https://raw.githubusercontent.com/guiwitz/MorphoDynamics/master/environment.yml) file to your computer (use ```Save as``` in your browser) and execute the following command from where you downloaded it:

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
pip install --upgrade git+https://github.com/guiwitz/MorphoDynamics.git
```

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

## Development

Versioning is done automatically via ```setuptools_scm```. To increment a version, create a new tag and push it to GitHub:

```
git tag -m "git versioning" -a X.Y.Z
git push origin X.Y.Z
```
