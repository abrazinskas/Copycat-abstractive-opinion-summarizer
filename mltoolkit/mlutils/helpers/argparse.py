"""Parameter parsing associated functions."""
from argparse import ArgumentTypeError


def str2bool(v):
    """Converts string to boolean."""
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise ArgumentTypeError('Boolean value expected.')


def str2list(v):
    """Converts a string  of comma separated values to a list of strs."""
    if len(v) == 0:
        return []
    return [str(item) for item in v.split(",")]


def str2intlist(v):
    """Converts a string  of comma separated values to a list of int."""
    if len(v) == 0:
        return []
    return [int(item) for item in v.split(",")]


def str2floatlist(v):
    """Converts a string of comma separated values to a list of float."""
    if len(v) == 0:
        return []
    return [float(item) for item in v.split(",")]
