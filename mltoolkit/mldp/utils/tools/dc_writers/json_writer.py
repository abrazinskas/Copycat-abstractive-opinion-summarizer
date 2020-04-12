from .base_writer import BaseWriter
from collections import OrderedDict
from mltoolkit.mlutils.helpers.general import listify
import json
from mltoolkit.mldp.utils.constants.dp import GROUPING_FNAMES
import numpy as np


class JsonWriter(BaseWriter):
    """
    Chunk writer specific for the JSON format. At the moment it permits only a
    single chunk to be written to a file. So one will be required to concatenate
    all chunks prior to writing in JSON.
    
    TODO: it violates the rule for list-JSON writing, do something about list-JSON
    TODO: writing to make it possible to dump chunks progressively.
    """

    def __init__(self, f, repr_funcs=None, grouping_fnames=None, indent=2):
        """
        :param f: an opened file where data-chunks should to be written.
        :param repr_funcs: dict of field names mapping to functions that
                           should be used to obtain str. reprs of field values.
        :param grouping_fnames: list of field names based on which data should
                                be grouped into a tree.
        :param indent: self-explanatory.
        """
        super(JsonWriter, self).__init__(f=f, repr_funcs=repr_funcs)
        self.grouping_fnames = listify(
            grouping_fnames) if grouping_fnames else None
        self.indent = indent

    def _exit(self, exc_type, exc_val, exc_tb):
        if not self.grouping_fnames:
            pass

    def _write(self, data_chunk):
        data_chunk.validate()
        if self.grouping_fnames:
            if self.f.tell() != 0:
                # TODO: give a better explanation of the problem.
                raise ValueError("Can't write multiple data-chunks in "
                                 "the json format.")
            self._write_as_tree(data_chunk)
        else:
            self._write_as_list(data_chunk)

    def _write_as_tree(self, data_chunk):
        """Writes the chunk by grouping on certain fields."""
        for gfn in self.grouping_fnames:
            if gfn not in data_chunk:
                raise ValueError("Grouping field '%s' is not present in the"
                                 " data-chunk." % gfn)
        other_fields = [fn for fn in data_chunk.keys() if
                        fn not in self.grouping_fnames]
        coll = OrderedDict()
        for indx in range(len(data_chunk)):
            group_vals = [data_chunk[fn][indx] for fn in self.grouping_fnames]
            cur_dict = coll

            for gv in group_vals:
                if gv not in cur_dict:
                    cur_dict[gv] = OrderedDict()
                cur_dict = cur_dict[gv]

            if not len(cur_dict):
                for fn in other_fields:
                    cur_dict[fn] = []

            for fn in other_fields:
                fv = data_chunk[fn][indx]
                rpr = self.repr_funcs[fn](fv) if fn in self.repr_funcs else fv
                cur_dict[fn].append(rpr)

        # adding the grouping field names as meta information
        # it will appear at the bottom of the file
        coll[GROUPING_FNAMES] = self.grouping_fnames
        json.dump(coll, self.f, indent=self.indent, default=convert)

    def _write_as_list(self, data_chunk):
        coll = []
        for du in data_chunk.iter():
            for k in du:
                if k in self.repr_funcs:
                    du[k] = self.repr_funcs[k](du[k])
            coll.append(du.to_dict())
        json.dump(coll, self.f, indent=self.indent, default=convert)


def convert(o):
    """Fixes a problem associated with the JSON not serializing numpy int64."""
    if isinstance(o, np.int64):
        return int(o)
    raise TypeError
