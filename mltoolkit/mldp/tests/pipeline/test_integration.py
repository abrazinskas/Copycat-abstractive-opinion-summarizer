import unittest
from mltoolkit.mldp.steps.readers import CsvReader
from mltoolkit.mldp.pipeline import Pipeline
from mltoolkit.mldp.steps.transformers.field import FieldSelector
from mltoolkit.mldp.steps.general import ChunkAccumulator
from mltoolkit.mldp.steps.formatters import PandasFormatter
from .comps import EmptyChunkTransformer
from mltoolkit.mldp.utils.constants.dp import EMPTY_CHUNK
from mltoolkit.mldp.utils.helpers.validation import equal_to_constant


class TestDataPipelineIntegration(unittest.TestCase):
    """
    Tests different integration aspects related to the data pipeline.
    Namely, how different combinations of steps operate together being wrapped
    by the data pipeline.
    """

    def test_simple_scenario(self):
        """
        Tries to run the pipeline, and if it works - it's considered to be
        successful. Tries different numbers of workers.
        """
        data_path = 'mldp/tests/data/small_chunks'
        field_names = ['first_name', 'email']
        worker_processes_nums = [0, 1, 2, 3, 4]

        reader = CsvReader(sep=",")

        for wpn in worker_processes_nums:

            dev_data_pipeline = Pipeline(reader=reader,
                                         worker_processes_num=wpn)
            dev_data_pipeline.add_step(FieldSelector(field_names))
            dev_data_pipeline.add_step(ChunkAccumulator(new_size=3))
            dev_data_pipeline.add_step(PandasFormatter())

            flag = False
            for data_chunk in dev_data_pipeline.iter(data_path=data_path):
                flag = True
                self.assertTrue(len(data_chunk) > 0)

        self.assertTrue(flag)

    def test_empty_chunks(self):
        """Testing whether empty chunks do not reach user."""
        data_path = 'mldp/tests/data/small_chunks'
        field_names = ['first_name', 'email']
        reader = CsvReader(chunk_size=1, sep=",")
        empty_chunk_transformer = EmptyChunkTransformer(max_count=3)

        dev_data_pipeline = Pipeline(reader=reader)
        dev_data_pipeline.add_step(empty_chunk_transformer)
        dev_data_pipeline.add_step(FieldSelector(field_names))

        flag = False
        for dc in dev_data_pipeline.iter(data_path=data_path):
            flag = True
            self.assertFalse(equal_to_constant(dc, EMPTY_CHUNK))

        self.assertTrue(flag)

    def test_readme_example(self):
        from mltoolkit.mldp.pipeline import Pipeline
        from mltoolkit.mldp.steps.readers import CsvReader
        from mltoolkit.mldp.steps.transformers.nlp import TokenProcessor, Padder
        from mltoolkit.mldp.steps.transformers.field import FieldSelector

        data_path = "mldp/tests/data/tweets.csv"

        # creating steps
        csv_reader = CsvReader(sep='\t', chunk_size=30)
        fields_selector = FieldSelector(fnames=["tweets", "labels"])
        token_processor = TokenProcessor(fnames="tweets",
                                         tokenization_func=lambda x: x.split(),
                                         lower_case=True)
        padder = Padder(fname="tweets", new_mask_fname="tweets_mask",
                        pad_symbol="<PAD>")

        # creating the pipeline
        pipeline = Pipeline(reader=csv_reader, worker_processes_num=1)
        pipeline.add_step(fields_selector)
        pipeline.add_step(token_processor)
        pipeline.add_step(padder)

        # iterate over data chunks
        for data_chunk in pipeline.iter(data_path=data_path):
            pass

        # generate documentation and print it
        print(pipeline)


if __name__ == '__main__':
    unittest.main()
