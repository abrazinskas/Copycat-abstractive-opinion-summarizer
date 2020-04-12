from mltoolkit.mldp.steps.formatters.base_formatter import BaseFormatter
import pandas


class PandasFormatter(BaseFormatter):
    """
    Converts chunk-chunks to pandas chunk-frames.
    """

    def _format(self, data_chunk):
        """
        :param data_chunk: self-explanatory.
        :return: Pandas DataFrame.
        """
        return pandas.DataFrame(data_chunk.data)
