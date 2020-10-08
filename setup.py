from setuptools import setup

version = {}
with open("morphodynamics/version.py") as fp:
    exec(fp.read(), version)

setup(name='morphodynamics',
      version=version["__version__"],
      description='Cell segmentation and windowing',
      url='',
      author='Cedric Vonesch and Guillaume Witz',
      author_email='',
      license='BSD3',
      packages=['morphodynamics'],
      zip_safe=False,
      install_requires=[
          'tifffile',
          'plotly',
          'aicsimageio',
          'nd2reader'
          ]
      )
