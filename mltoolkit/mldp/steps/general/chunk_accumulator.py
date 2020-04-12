from .base_general import BaseGeneral
from mltoolkit.mldp.steps.collectors import UnitCollector, \
    BaseChunkCollector


class ChunkAccumulator(BaseGeneral):
    """
    ChunkAccumulator step allows to group or change the size of data-chunks that
    are passed along the pipeline. The step does not alter the format of
    data-chunks only their size.

    For example, one might want to use larger chunks (e.g. size of 500) for
    computational purposes (fast vectorized operations on large numpy arrays)
    but to train a model on smaller data-chunks (e.g. size of 64). In that case,
    the step should be added after all computationally intensive ones.
    It works both by accumulating smaller upstream data-chunks and passing
    larger data-chunks downstream, and splitting larger upstream data-chunks
    into smaller downstream data-chunks.

    The adjuster uses chunk collectors, which have different notions of size.
    For example, UnitCollector works as described above. However, a
    more exotic collector can accumulate data-units which have the same 'id'
    field's value. And output a new chunk only after a sufficient number of
    unique ids is collected.
    """

    def __init__(self, collector=None, new_size=2, **kwargs):
        """
        :param collector: an object that accumulates data-chunks and yields
                          data-chunks when gets full.
        :param new_size: a parameters that is passed to the standard collector
                         object if collector is not provided.
        :param kwargs: self-explained.
        """
        super(ChunkAccumulator, self).__init__(**kwargs)

        if collector is None:
            self.coll = UnitCollector(max_size=new_size)
        else:
            if not isinstance(collector, BaseChunkCollector):
                raise TypeError("Please provide a valid collector that extends"
                                " the BaseChunkCollector class.")
            self.coll = collector

    def iter(self, data_chunk_iter):
        """
        Wraps the data-chunk iterable into a generator that yields data-chunks
        with the adjusted size.
        """
        # in case iteration was not performed until the end, reset the collector
        self.coll.reset()

        for data_chunk in data_chunk_iter:
            for adjusted_dc in self.coll.absorb_and_yield_if_full(data_chunk):
                yield adjusted_dc

        # yield the last (incomplete) chunk(s)
        for adjusted_dc in self.coll.yield_remaining():
            yield adjusted_dc

        self.coll.reset()

    def reset(self):
        self.coll.reset()
