#!/usr/bin/env python3

# SPDX-License-Identifier: GPL-2.0+ OR BSD-2-Clause

import logging

logger = logging.getLogger(__name__)
c_handler = logging.StreamHandler()
logger.addHandler(c_handler)
logger.setLevel(logging.DEBUG)
logger.propagate = False
