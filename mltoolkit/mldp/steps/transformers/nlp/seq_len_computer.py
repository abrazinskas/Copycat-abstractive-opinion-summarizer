from mltoolkit.mldp.steps.transformers import BaseTransformer
import numpy as np
from mltoolkit.mlutils.helpers.general import listify


class SeqLenComputer(BaseTransformer):
    """
    Computes lengths of sequences, stores them in a separate field.
    Assumes that field values support len() operation (e.g. lists, arrays, str).
    """

    def __init__(self, fname, new_len_fname, dtype='int64', **kwargs):
        super(SeqLenComputer, self).__init__(**kwargs)
        self.fnames = listify(fname)
        self.new_fnames = listify(new_len_fname)
        assert (len(self.fnames) == len(self.new_fnames))
        self.dtype = dtype

    def _transform(self, data_chunk):
        for fn, new_fn in zip(self.fnames, self.new_fnames):
            data_chunk[new_fn] = compute_lens(data_chunk[fn], dtype=self.dtype)
        return data_chunk


def compute_lens(fvalues, dtype='int64'):
    lens = np.zeros((len(fvalues),), dtype=dtype)
    for indx, fv in enumerate(fvalues):
        lens[indx] = len(fv)
    return lens
