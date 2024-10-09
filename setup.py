#!/usr/bin/env python
"""Scripts train SBAFs from VGAC to AVHRR."""

from setuptools import setup
from setuptools import find_packages

try:
    # HACK: https://github.com/pypa/setuptools_scm/issues/190#issuecomment-351181286
    # Stop setuptools_scm from including all repository files
    import setuptools_scm.integration
    setuptools_scm.integration.find_files = lambda _: []
except ImportError:
    pass


requires = ['satpy', 'tensorflow', 'numpy', 'netCDF4']


NAME = "sbafs_ann"
README = open('README.md', 'r').read()

setup(name=NAME,
      description='Tools to train SBAFs from VGAC to AVHRR',
      long_description=README,
      author='Nina Hakansson',
      author_email='nina.hakansson@smhi.se',
      classifiers=["Development Status :: 3 - Alpha",
                   "Intended Audience :: Science/Research",
                   "License :: OSI Approved :: GNU General Public License v3 " +
                   "or later (GPLv3+)",
                   "Operating System :: OS Independent",
                   "Programming Language :: Python",
                   "Topic :: Scientific/Engineering"],
      url="https://github.com/foua-pps/sbafs_ann",
      packages=find_packages(),
      scripts=["bin/train_sbafs.py",
               "bin/apply_sbafs.py"],
      data_files=[],
      zip_safe=False,
      use_scm_version=True,
      python_requires='>=3.7',
      install_requires=requires
      )
