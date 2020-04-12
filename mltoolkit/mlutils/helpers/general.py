import operator
import os
import psutil
import sys
import numpy as np
import inspect


def is_custom_object(obj):
    return hasattr(obj, '__dict__') or hasattr(obj, '__slots__')


def all_elements_are_equal(iterator):
    iterator = iter(iterator)
    try:
        first = next(iterator)
    except StopIteration:
        return True
    return all(first == rest for rest in iterator)


def flatten(my_list):
    """Flattens nested lists."""
    curr_items = []
    for x in my_list:
        if isinstance(x, list) and not isinstance(x, (str, bytes)):
            curr_items += flatten(x)
        else:
            curr_items.append(x)
    return curr_items


def sort_hash(hash, by_key=True, reverse=True):
    if by_key:
        indx = 0
    else:
        indx = 1
    return sorted(hash.items(), key=operator.itemgetter(indx), reverse=reverse)


def listify(val):
    """If val is an element the func wraps it into a list."""
    if isinstance(val, list):
        return val
    if isinstance(val, tuple):
        return list(val)
    return [val]


def ordered_dict_prepend(dct, key, value, dict_setitem=dict.__setitem__):
    root = dct._OrderedDict__root
    first = root[1]

    if key in dct:
        link = dct._OrderedDict__map[key]
        link_prev, link_next, _ = link
        link_prev[1] = link_next
        link_next[0] = link_prev
        link[0] = root
        link[1] = first
        root[1] = first[0] = link
    else:
        root[1] = first[0] = dct._OrderedDict__map[key] = [root, first, key]
        dict_setitem(dct, key, value)


def argsort(vals, order='ascending'):
    """Returns indices that can be used to produce sorted sequences."""
    if not isinstance(vals, (list, np.ndarray)):
        raise ValueError("Please provide a valid 'vals' param (list or array).")
    if order not in ['ascending', 'descending']:
        raise ValueError("Please provide a valid 'order' param"
                         " ('ascending' or 'descending').")
    if order == 'descending':
        return np.argsort(vals)[::-1]
    else:
        return np.argsort(vals)


def get_memory_usage():
    """Returns the memory usage of the current process in MB."""
    process = psutil.Process(os.getpid())
    return process.memory_info()[0] / 1e6


def get_size(obj, seen=None):
    """Recursively finds size of objects."""
    size = sys.getsizeof(obj)
    if seen is None:
        seen = set()
    obj_id = id(obj)
    if obj_id in seen:
        return 0
    # Important mark as seen *before* entering recursion to gracefully handle
    # self-referential objects
    seen.add(obj_id)
    if isinstance(obj, dict):
        size += sum([get_size(v, seen) for v in obj.values()])
        size += sum([get_size(k, seen) for k in obj.keys()])
    elif hasattr(obj, '__dict__'):
        size += get_size(obj.__dict__, seen)
    elif hasattr(obj, '__iter__') and not isinstance(obj,
                                                     (str, bytes, bytearray)):
        size += sum([get_size(i, seen) for i in obj])
    return size


def select_matching_kwargs(func, **kwargs):
    """
    Select key-value pairs from 'kwargs' where keys match expected arguments
    of 'func'. It's useful when 'kwargs' contains more arguments than a
    function expects, and one needs to strip the un-matching ones.
    
    :param func: function that expects certain arguments.
    :param kwargs: self-explanatory.
    :return: dict of matching arguments.
    """
    func_args = inspect.getargspec(func)[0]
    matching_args = {}
    for func_arg in func_args:
        if func_arg in kwargs:
            matching_args[func_arg] = kwargs[func_arg]
    return matching_args


def equal_to_constant(var, constant):
    if type(var) != type(constant):
        return False
    return var == constant


def merge_dicts(dct1, dct2):
    new_dict = {}
    for dct in [dct1, dct2]:
        for k, v in dct.items():
            assert k not in new_dict
            new_dict[k] = dct[k]
    return new_dict
