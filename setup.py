from setuptools import setup
from morphodynamics.version import get_version

version = get_version()

setup(name='morphodynamics',
      version=version,
      description='Cell segmentation and windowing',
      url='',
      author='Cedric Vonesch and Guillaume Witz',
      author_email='',
      license='BSD3',
      packages=['morphodynamics'],
      package_data={'morphodynamics': ['version.txt']},
      zip_safe=False,
      install_requires=[
          'tifffile',
          'plotly',
          'aicsimageio',
          'nd2reader'
          ]
      )
