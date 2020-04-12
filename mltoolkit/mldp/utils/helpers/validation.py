from numpy import isclose
import numpy as np
from collections import OrderedDict


def validate_field_names(field_names):
    """Checks whether field_names is either a string or a list of strings."""
    field_name_error_mess = "Please provide valid field_names." \
                            " It must be either a string or list of strings."
    if isinstance(field_names, list):
        for field_name in field_names:
            if not isinstance(field_name, str):
                raise ValueError(field_name_error_mess)
        return
    if not isinstance(field_names, str):
        raise ValueError(field_name_error_mess)


def validate_field_names_mapping(field_names_to_something, value_types):
    error_mess = "Please provide a valid mapping (dict, OrderedDict) from" \
                 " strings to "
    if isinstance(value_types, tuple):
        error_mess += ", ".join([v.__name__ for v in value_types])
    else:
        error_mess += value_types.__name__
    error_mess += " objects"
    if not isinstance(field_names_to_something, (dict, OrderedDict)):
        raise ValueError(error_mess)
    for key, val in field_names_to_something.items():
        if not isinstance(key, str) \
                or not isinstance(val, value_types):
            raise ValueError(error_mess)


def validate_data_paths(data_paths):
    """ Validates path(s) by checking their type. """
    if not isinstance(data_paths, (str, list)):
        raise ValueError("Please provide a valid data_path(s)")
    if isinstance(data_paths, list):
        for i, p in enumerate(data_paths):
            if not isinstance(p, str):
                raise ValueError("Data_path (%s) at the position %d is invalid"
                                 % (p, i))


def equal_to_constant(var, constant):
    if type(var) != type(constant):
        return False
    return var == constant


def equal_vals(val1, val2):
    """Recursively checks equality of two values."""
    if type(val1) != type(val2):
        return False
    if isinstance(val1, (list, tuple, np.ndarray)):
        if len(val1) != len(val2):
            return False
        for indx in range(len(val1)):
            if not equal_vals(val1[indx], val2[indx]):
                return False
        return True
    elif isinstance(val1, float):
        return isclose(val1, val2)
    else:
        return val1 == val2
