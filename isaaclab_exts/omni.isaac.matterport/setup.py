# Copyright (c) 2024 ETH Zurich (Robotic Systems Lab)
# Author: Pascal Roth
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Installation script for the 'omni.isaac.matterport' python package."""

from setuptools import find_namespace_packages, setup

INSTALL_REQUIRES = [
    "trimesh",
    "PyQt5",
    "matplotlib>=3.5.0",
    "pandas",
]

setup(
    name="omni-isaac-matterport",
    author="Pascal Roth",
    author_email="rothpa@ethz.ch",
    version="0.0.1",
    description="Extension to include Matterport 3D Datasets into Isaac (taken from https://niessner.github.io/Matterport/).",
    keywords=["robotics"],
    include_package_data=True,
    python_requires=">=3.10",
    install_requires=INSTALL_REQUIRES,
    packages=find_namespace_packages(include=["omni.isaac.matterport", "omni.isaac.matterport.*"]),
    classifiers=["Natural Language :: English", "Programming Language :: Python :: 3.11"],
    zip_safe=False,
)
