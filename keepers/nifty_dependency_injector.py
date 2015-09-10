# -*- coding: utf-8 -*-

import imp
import sys


class dependency_injector(object):
    def __init__(self, modules=[]):
        self.registry = {}
        map(self.register, modules)

    def get(self, x):
        return self.registry.get(x)

    def __getitem__(self, x):
        return self.registry.__getitem__(x)

    def __contains__(self, x):
        return self.registry.__contains__(x)

    def __iter__(self):
        return self.registry.__iter__()

    def __getattr__(self, x):
        return self.registry.__getattribute__(x)

    def register(self, module_name):
        if isinstance(module_name, tuple):
            module_name, key_name = (str(module_name[0]), str(module_name[1]))
        else:
            module_name = str(module_name)
            key_name = module_name
        try:
            loaded_module = sys.modules[module_name]
            self.registry[key_name] = loaded_module
        except KeyError:
            try:
                fp, pathname, description = imp.find_module(module_name)
                loaded_module = \
                    imp.load_module(module_name, fp, pathname, description)
                self.registry[key_name] = loaded_module
            except ImportError:
                pass
            finally:
                # Since we may exit via an exception, close fp explicitly.
                try:
                    fp.close()
                except (UnboundLocalError, AttributeError):
                    pass

