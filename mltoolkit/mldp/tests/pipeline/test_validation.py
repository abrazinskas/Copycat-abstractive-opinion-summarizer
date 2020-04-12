import unittest
from mltoolkit.mldp.steps.readers import CsvReader
from mltoolkit.mldp.pipeline import Pipeline
from mltoolkit.mldp.steps.transformers.field import FieldSelector
from mltoolkit.mldp.steps.transformers.general import FunctionApplier
from mltoolkit.mldp.steps.transformers.nlp import TokenProcessor
from mltoolkit.mldp.steps.general import ChunkAccumulator
from mltoolkit.mldp.steps.formatters import PandasFormatter
from mltoolkit.mldp.tests.pipeline.comps import InvalidCsvReader, \
    InvalidTransformer
from mltoolkit.mldp.utils.errors import DataChunkError
from itertools import permutations


class TestValidation(unittest.TestCase):

    def test_invalid_pipeline(self):
        """
        Tries to create an invalid data processing pipeline, and expect to get
        an error.
        """
        reader = CsvReader()
        with self.assertRaises(ValueError):
            data_pipeline = Pipeline(reader)
            data_pipeline.add_step(FieldSelector(["dummy"]))
            data_pipeline.add_step(PandasFormatter())
            data_pipeline.add_step(FunctionApplier({"dummy": lambda x: x}))

    def test_invalid_steps(self):
        """Testing whether an error is raised if an invalid step is present."""
        data_path = 'mldp/tests/data/news.csv'
        data_source = {'data_path': data_path}

        inv_reader = InvalidCsvReader()
        val_reader = CsvReader()

        val_transf1 = FieldSelector("text")
        val_transf2 = TokenProcessor(fnames='text')
        inv_transf1 = InvalidTransformer()
        accum = ChunkAccumulator(new_size=3)
        formatter = PandasFormatter()

        # try only the invalid reader and valid steps
        dp = Pipeline(reader=inv_reader, error_on_invalid_chunk='error')
        for vs in [val_transf1, val_transf2, accum, formatter]:
            dp.add_step(vs)
        with self.assertRaises(DataChunkError):
            for _ in dp.iter(**data_source):
                pass

        # try valid reader and invalid steps
        steps = [val_transf1, val_transf2, inv_transf1, accum]
        for st in permutations(steps):
            dp = Pipeline(reader=val_reader, error_on_invalid_chunk='error')
            for s in st:
                dp.add_step(s)
            dp.add_step(formatter)
            with self.assertRaises(DataChunkError):
                for _ in dp.iter(**data_source):
                    pass


if __name__ == '__main__':
    unittest.main()
