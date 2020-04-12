from mltoolkit.mldp.steps.readers import BaseReader


class DummyReader(BaseReader):
    def _iter(self, data_chunks):
        for dc in data_chunks:
            yield dc
