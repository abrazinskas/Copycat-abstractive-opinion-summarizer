import numpy as np


class BaseChunkCollector(object):
    """
    Children classes absorb data-chunks until get full, then yield a size
    adjusted data-chunk. Can be used for different kinds of groupings or
    data-units count adjustments.
    """

    def __init__(self, max_size=None):
        self.max_size = max_size

    def __len__(self):
        raise NotImplementedError

    def full(self):
        """Detects if the collector is full."""
        if self.max_size is None:
            return False
        return self.max_size <= len(self)

    def absorb_and_yield_if_full(self, data_chunk):
        """
        Adds the data-chunk to the collector, yields a new data_chunk if the
        collector is full.
        """
        raise NotImplementedError

    def yield_remaining(self):
        """
        After the upstream iteration is exhausted, this method allows to 
        yield remaining chunks, which might not be necessarily complete in
        terms of size.
        """
        raise NotImplementedError

    def _validate_input_length(self, value):
        if len(self) and len(value) != len(self):
            raise ValueError(
                "The size of the input value must be %d,"
                " got %d instead." % (len(self), len(value)))

    @staticmethod
    def _validate_input_value(value):
        if not isinstance(value, (list, np.ndarray)):
            raise TypeError(
                "The input value must be list or array, got '%s' instead." %
                type(value).__name__)

    def reset(self):
        raise NotImplementedError
