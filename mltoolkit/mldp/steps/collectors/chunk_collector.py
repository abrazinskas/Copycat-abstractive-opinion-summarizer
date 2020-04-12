from mltoolkit.mldp.steps.collectors import BaseChunkCollector
from mltoolkit.mldp.utils.helpers.dc import concat_chunks


class ChunkCollector(BaseChunkCollector):
    """
    Collects a fixed number of data-chunks, merges them and starts yielding."""

    def __init__(self, buffer_size, strict=True):
        """
        :param buffer_size: specifies how many chunks to collect before merging
                            and yielding is performed.
        :param strict: if set to True will not yield chunks that are incomplete.
        """
        super(ChunkCollector, self).__init__(max_size=buffer_size)
        self.strict = strict
        self._coll = None
        self.reset()

    def absorb_and_yield_if_full(self, data_chunk):
        self._coll.append(data_chunk)
        if self.full():
            merged_dc = self._merge_chunks()
            self.reset()
            yield merged_dc

    def __len__(self):
        return len(self._coll)

    def yield_remaining(self):
        if not self.strict:
            if len(self._coll):
                yield self._merge_chunks()

    def _merge_chunks(self):
        """Merges data-chunks and returns their generator."""
        merged_dc = concat_chunks(*[dc for dc in self._coll])
        return merged_dc

    def reset(self):
        self._coll = []
