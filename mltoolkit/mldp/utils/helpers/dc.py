import numpy as np
from mltoolkit.mldp.utils.tools import DataChunk


def concat_chunks(*dcs):
    """Combines data-chunks horizontally and returns them as one chunk."""
    new_dc = DataChunk()
    key_to_type = {}
    for dc in dcs:
        for k, v in dc.items():
            if k not in new_dc:
                new_dc[k] = []
            if isinstance(v, np.ndarray):
                if k in key_to_type and key_to_type[k] != np.ndarray:
                    raise TypeError("All values must either 'arrays' or "
                                    "'lists'.")
                key_to_type[k] = np.ndarray
                new_dc[k].append(v)
            elif isinstance(v, list):
                if k in key_to_type and key_to_type[k] != list:
                    raise TypeError("All values must either 'arrays' or "
                                    "'lists'.")
                key_to_type[k] = list
                new_dc[k] += v
            else:
                raise TypeError("Can't concat values other than 'lists' or "
                                "'arrays'.")

    for k in new_dc:
        if key_to_type[k] == np.ndarray:
            new_dc[k] = np.concatenate(tuple(new_dc[k]))
    return new_dc


def merge_chunks(dc_one, dc_two, merge_key):
    """Merges (vertically) chunks together by a 'merge_key'."""
    if not (dc_one[merge_key] == dc_two[merge_key]).all():
        raise ValueError("Can't merge chunks that have different fvalues.")

    new_dc = DataChunk()
    new_dc[merge_key] = dc_two[merge_key]

    for dc in [dc_one, dc_two]:
        for k in dc:
            if k != merge_key:
                new_dc[k] = dc[k]

    return new_dc
