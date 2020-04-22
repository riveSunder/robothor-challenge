from os.path import join, dirname, realpath
from setuptools import setup
import sys

setup(
    name="robothor",
    py_modules=["robothor_challenge"],
    version='0.1',
    install_requires=["numpy","ai2thor==2.3.7", \
            "pyyaml"],
    description="robothor contest development",
    author="Rive Sunder, AI2 Development Team",
    )








