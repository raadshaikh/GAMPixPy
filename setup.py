#!/usr/bin/env python

import setuptools

VER = "0.1"

reqs = ['numpy',
        'h5py',
        'tqdm',
        'torch',
        # 'ROOT',
        'particle',
        'yaml',
        ]

setuptools.setup(
    name="gampixpy",
    version=VER,
    author="SLAC National Accelerator Laboratory",
    author_email="dougl215@slac.stanford.edu",
    description="Gampix Readout Simulation",
    url="https://github.com/DanielMDouglas/GAMPixPy",
    packages=setuptools.find_packages(),
    install_requires=reqs,
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Topic :: Scientific/Engineering :: Physics"
    ],
    python_requires='>=3.2',
)
