import unittest
from mltoolkit.mldp.tests.common import generate_data_chunk, \
    create_list_of_data_chunks
from mltoolkit.mldp.steps.general import ChunkAccumulator
from mltoolkit.mldp.steps.collectors import ChunkShuffler
from mltoolkit.mldp.utils.helpers.dc import concat_chunks
import itertools
import numpy as np

np.random.seed(100)


class TestShuffledChunksCollector(unittest.TestCase):

    def test_order(self):
        """Testing production of chunks in a different order from the stream."""
        data_sizes = [200, 545]
        data_attrs_numbers = [5, 8, 2, 1, 15]
        inp_chunk_sizes = [1, 2, 3, 4, 5]
        buffer_sizes = [2, 38, 1000]

        for data_size, data_attrs_number, buffer_size, input_chunk_size in \
                itertools.product(data_sizes, data_attrs_numbers, buffer_sizes,
                                  inp_chunk_sizes):
            data = generate_data_chunk(data_attrs_number, data_size)
            inp_data_chunks = create_list_of_data_chunks(data, input_chunk_size)

            chunk_collector = ChunkShuffler(buffer_size=buffer_size)
            accum = ChunkAccumulator(collector=chunk_collector)

            actual_chunks = []

            for actual_chunk in accum.iter(inp_data_chunks):
                actual_chunks.append(actual_chunk)
            actual_ds = concat_chunks(*actual_chunks)

            self.assertTrue(data != actual_ds)
            self.assertTrue(len(data) == len(actual_ds))


if __name__ == '__main__':
    unittest.main()
