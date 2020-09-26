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
import entropart

setup(
        name='aisa',
        version=entropart.__version__,
        description=entropart.__description__,
        long_description=entropart.__long_description__,
        author=entropart.__author__,
        author_email=entropart.__author_email__,
        url=entropart.__url__,
        license=entropart.__copyright__,
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
