import unittest
from mltoolkit.mldp.tests.common import generate_data_chunk, \
    create_list_of_data_chunks
from mltoolkit.mldp.steps.general import ChunkAccumulator
from mltoolkit.mldp.steps.collectors import UnitCollector
import itertools


class ChunkAccumulatorTests(unittest.TestCase):

    def test_chunk_size_adjustment_with_random_data_and_params(self):
        """Default scenario when only the size of chunks is adjusted."""
        data_sizes = [100, 102, 54, 35]
        data_attrs_numbers = [5, 8, 2, 1, 15]
        inp_chunk_sizes = [10, 15, 63, 1, 2]
        batch_sizes = [1, 2, 38, 1000]

        for data_size, data_attrs_number, batch_size, input_chunk_size in \
                itertools.product(data_sizes, data_attrs_numbers, batch_sizes,
                                  inp_chunk_sizes):
            data = generate_data_chunk(data_attrs_number, data_size)
            expected_batches = create_list_of_data_chunks(data, batch_size)
            inp_data_chunks = create_list_of_data_chunks(data, input_chunk_size)

            chunk_collector = UnitCollector(max_size=batch_size)
            batcher = ChunkAccumulator(collector=chunk_collector)

            indx = 0
            for actual_chunk in batcher.iter(inp_data_chunks):
                expected_batch = expected_batches[indx]
                self.assertTrue(actual_chunk == expected_batch)
                indx += 1
            self.assertEqual(len(expected_batches), indx)


if __name__ == '__main__':
    unittest.main()
