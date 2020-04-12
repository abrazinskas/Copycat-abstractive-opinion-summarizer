from mltoolkit.mldp.steps.transformers import BaseTransformer
import numpy as np
from mltoolkit.mlutils.helpers.general import listify


class SeqWrapper(BaseTransformer):
    """
    Wraps each sequence with single start and end elements. Those are useful as
    indicators of segment/sentence beginning and ending for sequential models
    (e.g. LSTM).

    Allowed sequences types:
        1. Numpy arrays
        2. Lists
        2. String

    It explicitely avoid S-something arrays as output because it's hard to update
    its values. See example below.

    a = np.array(["a", "b"])
    a[0] = "AB"
    --> ["A", "b"]
    """

    def __init__(self, fname, start_el, end_el, **kwargs):
        """
        :param fname: the name or names(list) of the field(s) sequences of 
                           which to wrap.
        :param start_el: id or token of the start elem.
        :param end_el: id or token of the end elem.
        :param kwargs: e.g. name_prefix.
        """
        super(SeqWrapper, self).__init__(**kwargs)
        self.fname = listify(fname)
        self.start_el = start_el
        self.end_el = end_el

    def _transform(self, data_chunk):
        for fn in self.fname:
            data_chunk[fn] = wrap_seqs(data_chunk[fn], start=self.start_el,
                                       end=self.end_el)
        return data_chunk


def wrap_seqs(fvalues, start, end):
    """Wraps fvalues and returns either array of ints or objects."""
    if isinstance(fvalues, np.ndarray) and fvalues.dtype == np.int:
        res = [wrap_seq(seq, start=start, end=end) for seq in fvalues]
        res = np.array(res)
    elif isinstance(fvalues, list):
        res = [wrap_seq(fvalues[indx], start=start, end=end)
               for indx in range(len(fvalues))]
    else:
        res = np.empty(len(fvalues), dtype='object')
        for indx in range(len(fvalues)):
            res[indx] = wrap_seq(fvalues[indx], start=start, end=end)
    return res


def wrap_seq(seq, start, end):
    """Wraps a sequence (list, array, str) with start and end elems."""
    if isinstance(seq, list):
        seq = wrap_list(seq, start=start, end=end)
    elif isinstance(seq, np.ndarray):
        seq = wrap_numpy(seq, start=start, end=end)
    elif isinstance(seq, str):
        seq = wrap_str(seq, start=start, end=end)
    else:
        raise TypeError("All field values must be lists or numpy"
                        " arrays.")
    return seq


def wrap_list(seq, start, end):
    return [start] + seq + [end]


def wrap_numpy(seq, start, end):
    return np.concatenate([np.array([start]), seq, np.array([end])])


def wrap_str(seq, start, end):
    return '%s %s %s' % (start, seq, end)
