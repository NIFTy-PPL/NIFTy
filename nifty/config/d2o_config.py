# -*- coding: utf-8 -*-

import os

import keepers

# pre-create the D2O configuration instance and set its path explicitly
d2o_configuration = keepers.get_Configuration(
                    name='D2O',
                    file_name='D2O.conf',
                    search_pathes=[os.path.expanduser('~') + "/.config/nifty/",
                                   os.path.expanduser('~') + "/.config/",
                                   './'])
