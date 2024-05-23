#!/usr/bin/env python
from setuptools import setup

setup(
    name="d3m",
    version="1.0.0",
    description="D3M: Data Debiasing with Datamodels",
    long_description="",
    author="MadryLab",
    author_email="krisgrg@mit.edu",
    license_files=("LICENSE.txt",),
    packages=["d3m"],
    install_requires=["torch>=2.0.0", "numpy", "tqdm", "traker>=0.3.2"],
    include_package_data=True,
)
