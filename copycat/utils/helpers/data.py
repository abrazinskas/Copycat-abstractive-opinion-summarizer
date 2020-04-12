from collections import OrderedDict
import numpy as np


def group_vals_by_keys(vals, keys):
    """Returns an ordered dict that maps keys to values.

    :param vals: list.
    :param keys: list.
    """
    assert len(keys) == len(vals)
    map = OrderedDict()
    for val, key in zip(vals, keys):
        if key not in map:
            map[key] = []
        map[key].append(val)
    return map


def reverse_mapping(group_to_units, group_to_units_mask):
    """
    Reverses the mapping to unit => group indices. Where it's assumed
    that each unit is mapped to only one group.

    :return: unit_to_group: [units_count]
    """
    unit_count = int(group_to_units_mask.sum())
    assert unit_count == group_to_units.max() + 1
    unit_to_group = np.empty(unit_count, dtype=np.int64)
    for group_indx, (unit_indxs, unit_masks) in enumerate(zip(group_to_units,
                                                              group_to_units_mask)):
        _unit_indxs = unit_indxs[unit_masks == 1.]
        unit_to_group[_unit_indxs] = group_indx
    return unit_to_group


def get_all_but_one_rev_indxs(group_to_revs, group_to_revs_mask):
    """
    Computes all but one (the current) indexes for reviews. Assumes right
    padding and that there are no indxs of reviews missing.

    Returns reviews => other reviews mapping and mask, where target reviews are
    masked.
    """
    bs = int(group_to_revs_mask.sum())
    seq_len = group_to_revs.shape[1]
    rev_to_others = np.empty((bs, seq_len), dtype=np.int64)
    rev_to_others_mask = np.empty((bs, seq_len), dtype=np.float32)

    for rev_indxs, rev_masks in zip(group_to_revs, group_to_revs_mask):
        _rev_indxs = rev_indxs[rev_masks == 1.]
        rev_to_others[_rev_indxs] = rev_indxs[np.newaxis, :].repeat(
            (len(_rev_indxs)), 0)
        rev_to_others_mask[_rev_indxs] = rev_masks

        # masking the target ones
        rev_to_others_mask[_rev_indxs, np.arange(len(_rev_indxs))] = 0.

    return rev_to_others, rev_to_others_mask


def create_optimized_selector(other_rev_states_mask):
    """
    Compacts a mask of concatenated reviews belong to the same groups such that
    a selector would have less paddings, and thus would gain a computational
    speed-up.

    The produced indxs can be used to select non-masked concatenated review
    states that would produce more compact tensors.

    :param other_rev_states_mask: [batch_size, seq_len]
                                  where seq_len is the length of concat review
                                  states.
    :return: indxs: [batch_size, seq_len2]
             indxs_mask [batch_size, seq_len2]
    """
    bs = other_rev_states_mask.shape[0]
    local_len = other_rev_states_mask.sum(-1)
    max_non_zero = int(local_len.max())
    indxs = np.zeros((bs, max_non_zero), dtype=np.int64)
    indxs_mask = np.zeros((bs, max_non_zero), dtype=np.float32)

    for i in range(bs):
        orm = other_rev_states_mask[i]
        ll = int(local_len[i])
        indxs[i, :ll] = np.flatnonzero(orm)
        indxs_mask[i, :ll] = 1

    return indxs, indxs_mask
