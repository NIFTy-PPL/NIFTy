# -*- coding: utf-8 -*-

import ConfigParser
import pprint


class variable(object):
    def __init__(self, name, default_value_list, checker, genus='str'):
        self.name = str(name)
        if genus in ['str', 'int', 'float', 'boolean']:
            self.genus = str(genus)
        if not isinstance(default_value_list, list):
            raise ValueError("The default_value_list argument must be a list!")
        elif len(default_value_list) == 0:
            default_value_list = [None]
        self.default_value_list = default_value_list

        if callable(checker):
            self.checker = checker
        else:
            raise ValueError("The checker must be callable!")

        self.set_value(None)

    def set_value(self, value=None):
        if value is None:
            work_list = self.default_value_list
        else:
            work_list = [value]
        success = False
        for item in work_list:
            if self.checker(item):
                valid_item = item
                success = True
                break
        if not success:
            raise ValueError("No valid value supplied!" + str(work_list))
        else:
            self.value = valid_item
            return self.value

    def __call__(self):
        return self.get_value()

    def get_value(self):
        return self.value

    def get_name(self):
        return self.name

    def __repr__(self):
        return "<" + str(self.name) + "': " + \
               str(self.get_value()) + ">"


class configuration(object):
    """
    configuration
    init with file path to configuration file
    get command -> dictionary like
    set command -> parse input for sanity
    reset -> reset to defaults
    load from file
    save to file
    """
    def __init__(self, variables=[], path=None, path_section='DEFAULT'):
        self.variable_dict = {}
        map(self.register, variables)

        self.path = None
        self.set_path(path=path, path_section=path_section)
        try:
            self.load()
        except:
            pass

    def __getitem__(self, key):
        return self.get_variable(key)

    def __setitem__(self, key, value):
        return self.set_variable(key, value)

    def register(self, variable):
        self.variable_dict[variable.get_name()] = variable

    def get_variable(self, name):
        try:
            return self.variable_dict[name].get_value()
        except KeyError:
            raise KeyError("The requested variable  is not registered!")

    def set_variable(self, name, value):
        try:
            return self.variable_dict[name].set_value(value)
        except KeyError:
            raise KeyError("The requested variable  is not registered!")

    def set_path(self, path=None, path_section=None):
        if path is not None:
            self.path = str(path)
        if path_section is not None:
            self.path_section = str(path_section)

    def reset(self):
        for key, item in self.variable_dict.items():
            item.set_value(None)

    def validQ(self, name, value):
        return self.variable_dict[name].checker(value)

    def save(self, path=None, path_section=None):
        if path is None:
            if self.path is None:
                raise ValueError("No init- or keyword-path available.")
            else:
                path = self.path
        else:
            path = path

        if path_section is None:
            path_section = self.path_section

        config_parser = ConfigParser.ConfigParser()
        try:
            config_parser.add_section(path_section)
        except ValueError:
            pass

        for item in self.variable_dict:
            config_parser.set(path_section,
                              item,
                              str(self[item]))
        config_file = open(path, 'wb')
        config_parser.write(config_file)

    def load(self, path=None, path_section=None):
        if path is None:
            if self.path is None:
                raise ValueError("No init- or keyword-path available.")
            else:
                path = self.path
        else:
            path = path

        if path_section is None:
            path_section = self.path_section

        config_parser = ConfigParser.ConfigParser()
        config_parser.read(path)

        for key, item in self.variable_dict.items():
            if item.genus == 'str':
                temp_value = config_parser.get(path_section, item.name)
            elif item.genus == 'int':
                temp_value = config_parser.getint(path_section, item.name)
            elif item.genus == 'float':
                temp_value = config_parser.getfloat(path_section, item.name)
            elif item.genus == 'boolean':
                temp_value = config_parser.getboolean(path_section, item.name)
            else:
                raise ValueError("Unknown variable genus.")

            item.set_value(temp_value)

    def __repr__(self):
        return_string = ("<nifty configuration> \n" +
                         pprint.pformat(self.variable_dict))
        return return_string
















