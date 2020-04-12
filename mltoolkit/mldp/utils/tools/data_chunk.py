from collections import OrderedDict
import numpy as np
from mltoolkit.mldp.utils.helpers.validation import equal_vals
from mltoolkit.mldp.utils.errors import DataChunkError
from .data_unit import DataUnit
from mltoolkit.mldp.utils.tools.dc_writers import JsonWriter, CsvWriter
from copy import deepcopy


class DataChunk(object):
    """
    A collection of data-units that is passed along data pipeline.

    Valid data-chunk implies that it contains numpy arrays or lists of the same
    size, which enable an additional functionality:
        1. dumping/saving of data-chunks in the Json or CSV format.
        2. data-unit access.
    However, one can operate with invalid data-chunks for any reason,
    but previously mentioned functionality will be unavailable.
    """

    def __init__(self, **fields):
        """
        :param fields: names mapping to field values (not necessarily np.arrays).
        """
        self.data = OrderedDict()
        for fname, fval in fields.items():
            self.data[fname] = deepcopy(fval)

    @property
    def size(self):
        return len(self)

    @property
    def valid(self):
        return self._is_valid()

    @property
    def fnames(self):
        return list(self.keys())

    def items(self):
        return self.data.items()

    def keys(self):
        return self.data.keys()

    def values(self):
        return self.data.values()

    def append(self, data_unit):
        """
        Appends a new data-unit to the end of a valid data-chunk.

        :param data_unit: data unit or dict with field name and value pairs.
        """
        allowed_types = (dict, OrderedDict, DataUnit)
        rpr = [at.__name__ for at in allowed_types]
        if not isinstance(data_unit, allowed_types):
            raise TypeError("'data_unit' must be %s." % " or ".join(rpr))
        if not self._is_valid():
            raise DataChunkError("Can't append a new data-unit to an "
                                 "invalid data-chunk.")
        if len(self.fnames) > 0:
            for k in data_unit:
                if k not in self:
                    raise ValueError(
                        "Please provide all keys matching existing "
                        "field names.")
        else:
            for k in data_unit:
                if isinstance(data_unit, DataUnit):
                    ds_type = data_unit.ds_type(k)
                    if ds_type == list:
                        ds = []
                    elif ds_type == np.ndarray:
                        ds = np.array([])
                    else:
                        raise TypeError("Can't handle '%s' type." % ds_type)
                else:
                    ds = []
                self[k] = ds

        for k in data_unit:
            if isinstance(self[k], np.ndarray):
                self[k] = np.append(self[k], data_unit[k])
            elif isinstance(self[k], list):
                self[k].append(data_unit[k])
            else:
                raise NotImplementedError

    def iter(self):
        """Creates a data-units generator."""
        if not self._is_valid():
            raise DataChunkError("Can't iterate over an invalid data-chunk.")
        for indx in range(len(self)):
            data_unit = self[indx]
            yield data_unit

    def _is_valid(self):
        try:
            self.validate()
        except Exception as e:
            return False
        return True

    def validate(self):
        """Checks if all field values are lists/arrays of the same length."""
        not_arr_error_mess = "Data-chunk field values must be numpy arrays " \
                             "or lists, while '%s' field contains: '%s'."
        not_same_len_error_mess = "All data-chunk field value arrays/lists " \
                                  "must be of the same size."
        prev_len = None
        for k, v in self.items():
            if not isinstance(v, (list, np.ndarray)):
                raise DataChunkError(not_arr_error_mess % (k, type(v).__name__))
            curr_len = len(v)
            if prev_len is not None and prev_len != curr_len:
                raise DataChunkError(not_same_len_error_mess)
            prev_len = curr_len

    def __iter__(self):
        return iter(self.keys())

    def __getitem__(self, *args):
        """
        Permits access of data-units, field values, or a specific field value.

        1. In order to access data-units, provide an integer indx, e.g.:
            --> dc[indx]
           Will return a special object that allows easy access to the specific
           data-unit's field values. Throws an error if the data-chunk is
           invalid.

        2. To access all field values (the whole column) provide a str
           field name, e.g.: --> dc[fname]
           Will return the content of the field, e.g. numpy array or list if the
           data-chunk is valid.

        3. To access a specific field value provide an integer indx, and string
            field name, e.g. --> dc[indx, fname]
        """
        items = args[0]
        if not isinstance(items, tuple):
            item = items
            # return field values
            if isinstance(item, str):
                return self.data[item]
            # return a data-unit
            if isinstance(item, int):
                self.validate()
                return DataUnit(fnames=self.fnames, data=self.data, indx=item)
            return TypeError("Please provide 'str' argument to "
                             "access field values, or 'int' to access "
                             "a specific data-unit.")
        elif len(items) == 2:
            indx, fname = items
            if not isinstance(indx, (int, np.int64, np.int32)):
                raise TypeError("Please provide an 'int' indx as the first "
                                "argument.")
            if not isinstance(fname, str):
                raise TypeError("Please provide a 'str' field "
                                "name as the second argument.")
            return self.data[fname][indx]
        else:
            raise ValueError("Invalid access of the data-chunk. Please access "
                             "as follow: dc[indx], dc[fname], or "
                             "dc[indx, fname].")

    def to_csv(self, f, repr_funcs=None, sep='\t'):
        """
        Dumps the chunk to a csv file.

        :param f: an opened file where the data-chunk should to be written.
        :param repr_funcs: dict of field names mapping to functions that
                           should be used to obtain str. reprs of field values.
        :param sep: separator of values.
        """
        with CsvWriter(f=f, repr_funcs=repr_funcs, sep=sep) as w:
            w.write(self)

    def to_json(self, f, repr_funcs=None, grouping_fnames=None, indent=2):
        """
        Dumps the chunk to a json file optionally by grouping on certain fields.

        :param f: an opened file where the data-chunk should to be written.
        :param repr_funcs: dict of field names mapping to functions that
                           should be used to obtain str. reprs of field values.
        :param grouping_fnames: list of field names based on which data should
                                be grouped into a tree.
        :param indent: self-explanatory.
        """
        with JsonWriter(f=f, repr_funcs=repr_funcs, indent=indent,
                        grouping_fnames=grouping_fnames) as w:
            w.write(self)

    def __contains__(self, key):
        return key in self.data

    def __delitem__(self, key):
        """
        Deletion of a complete field or a data-unit. If a string is
        passed then the former is performed, if an integer than the latter.
        Also, the former performs validation of the data-chunk's validity.
        """
        if isinstance(key, int):
            if not self.valid:
                raise ValueError("Can't delete from an invalid data-chunk.")
            for fname in self.fnames:
                if isinstance(self.data[fname], np.ndarray):
                    self.data[fname] = np.delete(self.data[fname], key, axis=0)
                else:
                    del self.data[fname][key]
        elif isinstance(key, str):
            del self.data[key]
        else:
            raise TypeError("Please provide 'int', 'str' as a key.")

    def __setitem__(self, key, value):
        """
        Multi-functional method that allows to set:
            1. a field's value (provide key:str, values:arbitrary
            2. a data-unit's values (provide key:int, value:dict/o_dict)
            3. a field's values (provide key:str, values:arbitrary)
        """
        if isinstance(key, tuple):
            # setting a field's value
            assert len(key) == 2
            indx, fname = key
            if not isinstance(indx, int):
                raise TypeError("Please provide an 'int' indx as the first "
                                "argument.")
            if not isinstance(fname, str):
                raise TypeError("Please provide a 'str' field "
                                "name as the second argument.")
            self.data[fname][indx] = value
        else:
            if isinstance(key, int):
                # setting a data-unit
                self.validate()
                if not isinstance(value, (dict, OrderedDict)):
                    raise TypeError("When setting a data-unit. The provided "
                                    "'value' argument must be a 'dict'.")
                for k in value.keys():
                    if k not in self:
                        raise ValueError("Please provide all keys matching "
                                         "existing field names.")
                for k, v in value.items():
                    self.data[k][key] = v
            else:
                # setting a field's values
                self.data[key] = value

    def __eq__(self, other):
        if not isinstance(other, DataChunk):
            return False

        if len(self.keys()) != len(other.keys()):
            return False

        for k in self.keys():
            if k not in other:
                return False
            if not equal_vals(self[k], other[k]):
                return False

        return True

    def __len__(self):
        """
        It's important to note that no guarantees are provided for invalid
        data-chunks. Validation is removed for convenience.
        """
        # self.validate()
        if len(self.fnames):
            return len(self[self.fnames[0]])
        return 0

    def __str__(self):
        return str(self.data.items())
