"""
PEP 517 doesn’t support editable installs
so this file is currently here to support "pip install -e ."
"""
from setuptools import setup

use_scm={"write_to": "morphodynamics/version.py"}
setup(use_scm_version=use_scm)