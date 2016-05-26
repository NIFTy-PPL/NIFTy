# -*- coding: utf-8 -*-

import os

import keepers

# pre-create the D2O configuration instance and set its path explicitly
d2o_configuration = keepers.get_Configuration(
                        'D2O',
                        path=os.path.expanduser('~') + "/.nifty/d2o_config")
