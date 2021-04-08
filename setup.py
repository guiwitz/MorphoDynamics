from setuptools import setup, find_packages
from morphodynamics.version import get_version

#version = get_version()
use_scm = {"write_to": "morphodynamics/_version.py"}

setup(name='morphodynamics',
      #version=version,
      description='Cell segmentation and windowing',
      url='',
      author='Cedric Vonesch and Guillaume Witz',
      author_email='',
      license='BSD3',
      packages=find_packages(),
      package_data={'morphodynamics': ['version.txt']},
      zip_safe=False,
      use_scm_version=use_scm,
      install_requires=[
          'numpy',
          'pandas',
          'matploltib',
          'scikit-image',
          'tifffile',
          'h5py',
          'plotly',
          'aicsimageio',
          'setuptools_scm',
          'dask[complete]',
          'dask-jobqueue',
          'nd2reader@git+https://github.com/guiwitz/nd2reader.git@master#egg=nd2reader'
          ]
      )
