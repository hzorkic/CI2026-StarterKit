#!/bin/env python
# -*- coding: utf-8 -*-
#
# Built for the CI 2026 hackathon starter kit

# System modules

# External modules

# Internal modules

from setuptools import find_packages, setup

setup(
    name="ci_2026_starter_kit",
    packages=find_packages(
        include=["starter_kit", "internal", "data_preparation", "data"]
    ),
    version="0.1",
    description="The starter kit for the CI 2026 hackathon",
    author="Tobias S. Finn",
    license="MIT",
)
