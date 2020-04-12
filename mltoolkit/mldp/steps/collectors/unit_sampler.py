from mltoolkit.mldp.steps.collectors import BaseChunkCollector
from mltoolkit.mldp.utils.tools import DataChunk
from numpy import random
from collections import OrderedDict
from mltoolkit.mldp.utils.errors import DataChunkError
import numpy as np


class UnitSampler(BaseChunkCollector):
    """Groups data-units based on `id_fname`, produces samples of data-units.

    Sequentially absorbs data-chunks until a data-unit with a different
    `id_fname` entry is encountered. In such case, it will start the sampling
    phase.

    If a group has more units than the maximum specified, than the
    specified number of units will be sampled to create each chunk until
    all units are used. Otherwise, sampling is performed only once.
    """

    def __init__(self, id_fname, min_units=None, max_units=None,
                 sample_all=True):
        """
        :param id_fname: self-explanatory.
        :param min_units: if a batch or a group has less reviews than
                                  specified, it is discarded.
        :param max_units: the limit of reviews that each group
                                  in the chunk will have. When limit is
                                  reached for a group, the other reviews
                                  are ignored.
        :param sample_all: if set to True performs sampling of reviews until
                                all are used, otherwise samples only once and
                                drops the rest.
        """
        super(UnitSampler, self).__init__(max_size=1)
        self.id_fname = id_fname
        if min_units and max_units:
            assert min_units <= max_units
        self.min_units = min_units
        self.max_units = max_units
        self.sample_all_revs = sample_all

        self._coll = OrderedDict()
        self._prev_group_id = None

    def __len__(self):
        first_key = next(iter(self._coll.keys()))
        return len(self._coll[first_key])

    def absorb_and_yield_if_full(self, data_chunk):
        for indx in range(len(data_chunk)):
            group_id = data_chunk[indx, self.id_fname]

            if self._prev_group_id and group_id != self._prev_group_id:
                for chunk in self.yield_remaining():
                    yield chunk
                self.reset()
            self._prev_group_id = group_id

            if not len(self._coll):
                for fn in data_chunk.fnames:
                    if isinstance(data_chunk[fn], np.ndarray):
                        self._coll[fn] = np.array([],
                                                  dtype=data_chunk[fn].dtype)
                    else:
                        self._coll[fn] = []

            for fn in data_chunk.fnames:
                val = data_chunk[indx, fn]
                if fn not in self._coll:
                    raise DataChunkError("Input chunks have different field "
                                         "names.")
                if isinstance(self._coll[fn], np.ndarray):
                    self._coll[fn] = np.append(self._coll[fn], val)
                else:
                    self._coll[fn].append(val)

    def yield_remaining(self):
        for chunk in self.compile_chunks():
            if self.min_units and len(chunk) < self.min_units:
                continue
            yield chunk

    def get_coll_len(self):
        if not len(self._coll):
            return 0
        first_key = next(iter(self._coll.keys()))
        return len(self._coll[first_key])

    def reset(self):
        self._coll = OrderedDict()
        self._prev_group_id = None

    def compile_chunks(self):
        """Compiles data-chunks filled with group sequences."""
        if self.max_units:
            while len(self):
                if len(self) > self.max_units:
                    sel_indxs = random.choice(range(len(self)), replace=False,
                                              size=self.max_units)
                else:
                    sel_indxs = range(len(self))

                # create an output data-chunk based on the selected units
                dc = DataChunk()
                for k, val in self._coll.items():
                    dc[k] = [val[indx] for indx in sel_indxs]
                    if isinstance(val, np.ndarray):
                        dc[k] = np.array(dc[k], dtype=val.dtype)
                yield dc

                # removing the selected indxs from the collector
                for indx in sorted(sel_indxs, reverse=True):
                    for fn in self._coll:
                        if isinstance(self._coll[fn], np.ndarray):
                            self._coll[fn] = np.delete(self._coll[fn], indx)
                        else:
                            del self._coll[fn][indx]

                # stop the cycle as one sample is already produced
                if not self.sample_all_revs:
                    break
        else:
            dc = DataChunk()
            for k, val in self._coll.items():
                dc[k] = val
            yield dc
