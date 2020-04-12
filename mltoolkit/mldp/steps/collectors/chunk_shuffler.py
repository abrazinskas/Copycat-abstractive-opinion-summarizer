from mltoolkit.mldp.steps.collectors import BaseChunkCollector
from numpy import random


class ChunkShuffler(BaseChunkCollector):
    """
    Collects a fixed number of data-chunks, shuffles them, than starts to
    yield one by one.

    One application of this collector is post-grouping shuffling. In other 
    words, one first groups data-units by some group, and then shuffles them.
    """

    def __init__(self, buffer_size=None):
        """
        :param buffer_size: defines how many chunks to collect before shuffling
                            is performed and chunks start to get yield.
        """
        super(ChunkShuffler, self).__init__(max_size=buffer_size)
        self._coll = []

    def absorb_and_yield_if_full(self, data_chunk):
        self._coll.append(data_chunk)
        if self.full():
            for dc in self._create_chunk_gen():
                yield dc
            self.reset()

    def __len__(self):
        return len(self._coll)

    def yield_remaining(self):
        for dc in self._create_chunk_gen():
            yield dc

    def _create_chunk_gen(self):
        """Shuffles data-chunks and returns their generator."""
        self._shuffle_coll()
        for dc in self._coll:
            yield dc

    def reset(self):
        self._coll = []

    def _shuffle_coll(self):
        """Shuffles the collector with data-chunks."""
        indxs = random.permutation(len(self))
        new_coll = [self._coll[indx] for indx in indxs]
        self._coll = new_coll
