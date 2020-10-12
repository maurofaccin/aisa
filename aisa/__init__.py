#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""

.. include:: ../README.md
"""

# Author: Mauro Faccin 2020
# -------------------------
# |   This is AISA        |
# -------------------------
# |    License: GPL3      |
# |   see LICENSE.txt     |
# -------------------------


from .base import *

__all__ = ["PGraph", "best_partition", "merge_pgraph", "entrogram", "optimize"]

__productname__ = 'AISA'
__version__ = '0.1'
__copyright__ = "Copyright (C) 2020 Mauro Faccin"
__author__ = "Mauro Faccin"
__author_email__ = "mauro.fccn@gmail.com"
__description__ = "AISA: Auto-Information State Aggregation"
__long_description__ = "State aggregation through maximization of the auto-information of the dynamics"
__url__ = "https://maurofaccin.github.io"
__license__ = "Licensed under the GNU GPL v3+."
