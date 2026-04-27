"""Installation script for the 'omni.isaac.vlnce' python package."""

from setuptools import find_namespace_packages, setup

INSTALL_REQUIRES = [
    "numpy",
    "scipy>=1.7.1",
    "torch>=2.2.0",
]

setup(
    name="omni-isaac-vlnce",
    version="0.0.1",
    keywords=["robotics", "navigation"],
    include_package_data=True,
    python_requires=">=3.10",
    install_requires=INSTALL_REQUIRES,
    packages=find_namespace_packages(include=["omni.isaac.vlnce", "omni.isaac.vlnce.*"]),
    classifiers=["Natural Language :: English", "Programming Language :: Python :: 3.11"],
    zip_safe=False,
)
