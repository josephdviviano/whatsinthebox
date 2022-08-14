#!/usr/bin/env python

from distutils.core import setup

from setuptools import setup, find_packages

setup(
    name='witb',
    version='0.0.1',
    packages=find_packages(include=['witb', 'witb.*']),
    python_requires='>=3.6',
    entry_points={
        'console_scripts': [
            'main=witb.main:main'
        ],
    }
)
