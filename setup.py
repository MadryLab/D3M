#!/usr/bin/env python
from setuptools import setup

setup(
    name="dda",
    version="1.0.0",
    description="DDA: Debiasing Through Data Attribution",
    long_description="",
    author="MadryLab",
    author_email="krisgrg@mit.edu",
    license_files=("LICENSE.txt",),
    packages=["dda"],
    install_requires=[
        "torch>=2.0.0",
        "numpy",
        "tqdm",
        "trak>=0.3.2"
    ],
    include_package_data=True,
)
