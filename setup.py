#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# Author: Mauro Faccin 2020
# -------------------------
# |   This is AISA        |
# -------------------------
# |    License: GPL3      |
# |   see LICENSE.txt     |
# -------------------------

try:
    from setuptools import setup
except ImportError:
    from distutils.core import setup
import aisa

setup(
        name='aisa',
        version=aisa.__version__,
        description=aisa.__description__,
        long_description=aisa.__long_description__,
        author=aisa.__author__,
        author_email=aisa.__author_email__,
        url=aisa.__url__,
        license=aisa.__copyright__,
        packages=['aisa'],
        requires=[
            'numpy',
            'scipy',
            'networkx',
            ],
        provides=['aisa'],
        classifiers=[
            'Development Status :: 4 - Beta',
            'Environment :: Console',
            'Intended Audience :: End Users/Desktop',
            'Intended Audience :: Science/Research',
            'Intended Audience :: Information Technology',
            'License :: OSI Approved :: GNU General Public License v3 (GPLv3)',
            'Operating System :: OS Independent',
            'Programming Language :: Python',
            'Topic :: Scientific/Engineering :: Physics'
            ],
        )
