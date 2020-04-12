from mltoolkit.mldp.steps import BaseStep
import types


class BaseReader(BaseStep):
    """
    Defines the blue-print for children classes that read data from a local and
    remote storage.

    Reader objects can be used as standalone objects for raw data-chunks
    iteration, simply use .iter(**data_source_params). Alternatively, it can be
    used in a pipeline.
    """

    def __init__(self, chunk_size=None, **kwargs):
        super(BaseReader, self).__init__(**kwargs)
        self.chunk_size = chunk_size

    def iter(self, **kwargs):
        """
        Creates an iterable generator of data chunks which are created by
        reading data specified by **kwargs.

        :param kwargs: must be coherent with _iter method that is implemented in
                       subclasses. E.g. it might data_path, along with
                       units_per_file that control the units that are read from
                       each file.
        :return: generator of data-chunks.
        """
        itr = self._iter(**kwargs)
        if not isinstance(itr, types.GeneratorType):
            raise ValueError("_iter method needs to return a generator.")
        return itr

    def _iter(self, **kwargs):
        """
        One has to implement the actual logic for data reading in subclass
        readers.

        :return: generator of data-chunks.
        """
        raise NotImplementedError
