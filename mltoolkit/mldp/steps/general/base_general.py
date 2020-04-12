from mltoolkit.mldp.steps import BaseStep


class BaseGeneral(BaseStep):
    def iter(self, data_chunk_iter):
        """Inputs an iterable and outputs a generator."""
        raise NotImplementedError
