from collections import OrderedDict


class DataUnit(object):
    """
    Provides an access abstraction over specific data-chunk records.
    Does not perform copying of data, but rather operates by reference.

    Objects of this class should not be initialized manually but only internally
    from the DataChunk class.
    """

    def __init__(self, fnames, data, indx):
        self._fnames = fnames
        self._data = data
        self.indx = indx

    @property
    def fnames(self):
        return self._fnames

    def keys(self):
        return self.fnames

    def values(self):
        for fname in self.fnames:
            yield self[fname]

    def items(self):
        for fname in self.fnames:
            yield fname, self[fname]

    def to_dict(self):
        od = OrderedDict()
        for k, v in self.items():
            od[k] = v
        return od

    def ds_type(self, fname):
        return type(self._data[fname])

    def __iter__(self):
        return iter(self.fnames)

    def __getitem__(self, fname):
        return self._data[fname][self.indx]

    def __setitem__(self, fname, fvalue):
        self._data[fname][self.indx] = fvalue

    def __eq__(self, other):
        if not isinstance(other, DataUnit):
            return False

        if len(other) != len(self):
            return False

        for k, v in self.items():
            if k not in other:
                return False
            if v != other[k]:
                return False

        return True

    def __contains__(self, item):
        return item in self.keys()

    def __len__(self):
        return len(self.keys())

    def __str__(self):
        return "{" + ", ".join(["%s: %s" % (fname, self[fname]) for fname in
                                self.fnames]) + "}"
