from codecs import StreamReaderWriter, StreamWriter
from io import IOBase


class BaseWriter(object):
    """
    Children classes of this class contain format specific logic for writing
    data-chunks to the storage.
    """

    def __init__(self, f, repr_funcs=None):
        """
        :param f: an opened file where data-chunks should to be written.
        :param repr_funcs: dict of field names mapping to functions that
                           should be used to obtain str. reprs of field values.
        """
        if not isinstance(f, (StreamWriter, StreamReaderWriter, IOBase)):
            raise TypeError("Please provide an opened file for writing.")
        self.f = f
        self.repr_funcs = repr_funcs if repr_funcs else {}

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self._exit(exc_type, exc_val, exc_tb)
        self.f.close()

    def write(self, data_chunk):
        # if not isinstance(data_chunk, DataChunk):
        #     raise ValueError("Please provide a data-chunk object as argument "
        #                      "'data_chunk'.")
        data_chunk.validate()
        self._write(data_chunk)

    def _exit(self, exc_type, exc_val, exc_tb):
        raise NotImplementedError

    def _write(self, data_chunk):
        raise NotImplementedError
