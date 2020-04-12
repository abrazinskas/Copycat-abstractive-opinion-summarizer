from mltoolkit.mldp.steps.transformers import BaseTransformer
from mltoolkit.mldp.utils.constants.dp import EMPTY_CHUNK


class EmptyChunkTransformer(BaseTransformer):
    """Passes a fixed number of empty chunks, rest are passed untouched."""

    def __init__(self, max_count, **kwargs):
        super(EmptyChunkTransformer, self).__init__(**kwargs)
        self.max_count = max_count
        self.current_count = 0

    def _transform(self, data_chunk):
        self.current_count += 1
        if self.current_count <= self.max_count:
            return EMPTY_CHUNK
        return data_chunk
