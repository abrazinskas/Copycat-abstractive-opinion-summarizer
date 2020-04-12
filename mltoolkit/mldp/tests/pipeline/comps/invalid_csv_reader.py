from mltoolkit.mldp.steps.readers import CsvReader


class InvalidCsvReader(CsvReader):
    """This reader yields dicts of numpy arrays instead of data-chunks."""

    def __init__(self, **kwargs):
        super(InvalidCsvReader, self).__init__(**kwargs)

    def _iter(self, data_path):
        for dc in CsvReader._iter(self, data_path=data_path):
            yield dc.data
