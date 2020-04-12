import torch as T
from torch import Tensor
import numpy as np


def optimize_att_tens(mask, *tens):
    """
    Optimizes concatenated seqs for attention by removing non-zero ones,
    and re-padding.
    TODO: optimize it to avoid the for-loop!

    :param tens: list of [batch_size, seq_len, dim]
    :param mask: [batch_size, seq_len]
    :return: new_tens: [bs, new_seq_len, dim]
             new_mask: [bs, new_seq_len]
    """
    bs = tens[0].size(0)
    device = tens[0].device

    non_zeros = mask.sum(-1).long()
    max_non_zero = non_zeros.max()

    new_tens = [T.zeros((bs, max_non_zero, ten.size(2)),
                        dtype=ten.dtype, device=device) for ten in tens]
    new_mask = T.zeros((bs, max_non_zero), dtype=mask.dtype, device=device)

    for indx in range(bs):
        curr_mask = mask[indx]
        curr_non_zeros = non_zeros[indx]
        new_mask[indx, :curr_non_zeros] = 1

        for i, ten in enumerate(tens):
            curr_unit = tens[i][indx]
            new_tens[i][indx, :curr_non_zeros] = curr_unit[curr_mask == 1.]

    return new_tens, new_mask


def group_att_over_input(inp_att_vals, inp_att_mask, att_indxs, att_indxs_mask,
                         inp_att_keys=None):
    """
    Returns properly concatenated and shaped attention keys/values/mask over
    input (e.g. hidden states and embds).

    :param inp_att_keys: [inp_batch_size, seq_len, att_hidden_dim]
    :param inp_att_vals: [inp_batch_size, seq_len, val_dim]
    :param inp_att_mask: [inp_batch_size, seq_len]
    :param att_indxs: [out_batch_size, els_per_unit]
    :param att_indxs_mask: [out_batch_size, els_per_unit]
    :return: att_keys: [out_batch_size, total_els, att_hidden_dim]
             att_values: [out_batch_size, total_els, val_dim]
             att_mask: [out_batch_size, total_els]
    """
    seq_len = inp_att_vals.size(1)
    out_batch_size = att_indxs.size(0)
    els_per_unit = att_indxs.size(1)
    total_els_per_unit = els_per_unit * seq_len

    inp_att_vals = inp_att_vals[att_indxs] * \
                   att_indxs_mask.unsqueeze(-1).unsqueeze(-1)
    inp_att_mask = inp_att_mask[att_indxs] * \
                   att_indxs_mask.unsqueeze(-1)

    att_vals = inp_att_vals.view(out_batch_size, total_els_per_unit, -1)
    att_mask = inp_att_mask.view(out_batch_size, total_els_per_unit)

    if inp_att_keys is not None:
        inp_att_keys = inp_att_keys[att_indxs] * \
                       att_indxs_mask.unsqueeze(-1).unsqueeze(-1)
        att_keys = inp_att_keys.view(out_batch_size, total_els_per_unit, -1)
        return att_keys, att_vals, att_mask
    else:
        return att_vals, att_mask


def get_avg_repr(inp_reprs, indxs, indxs_mask):
    """
    Returns the average representation based on passed indxs and mask.

    :param inp_reprs: [batch_size1, dim]
    :param indxs: [batch_size2, seq_len]
    :param indxs_mask: [batch_size2, seq_len]
    :return: [batch_size2, dim]
    """
    sel_reprs = inp_reprs[indxs]  # [batch_size2, seq_len, dim]
    avg_reprs = (sel_reprs * indxs_mask.unsqueeze(-1)).sum(dim=1)
    avg_reprs = avg_reprs / indxs_mask.sum(-1, keepdim=True).float()
    return avg_reprs
