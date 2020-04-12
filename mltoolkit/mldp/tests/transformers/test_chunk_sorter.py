import unittest
from mltoolkit.mldp.steps.transformers.general import ChunkSorter
from mltoolkit.mldp.utils.tools import DataChunk
import numpy as np


class TestChunkSorter(unittest.TestCase):
    def setUp(self):
        self.ints_fn = "ints"
        self.strings_fn = "strings"
        self.floats_fn = "floats"

    def test_sorting_by_ints_descending(self):
        expected_dc = DataChunk(**{
            self.ints_fn: np.array([123, 10, 0]),
            self.strings_fn: np.array(["d", "a", "c"]),
            self.floats_fn: np.array([15., -1, -10.])
        })
        actual_dc = self._run_sorter(fn=self.ints_fn, order='descending')
        self.assertTrue(expected_dc == actual_dc)

    def test_sorting_by_ints_ascending(self):
        expected_dc = DataChunk(**{
            self.ints_fn: np.array([0, 10, 123]),
            self.strings_fn: np.array(["c", "a", "d"]),
            self.floats_fn: np.array([-10., -1., 15.])
        })
        actual_dc = self._run_sorter(fn=self.ints_fn, order='ascending')
        self.assertTrue(expected_dc == actual_dc)

    def test_sorting_by_strings_descending(self):
        expected_dc = DataChunk(**{
            self.ints_fn: np.array([123, 0, 10]),
            self.strings_fn: np.array(["d", "c", "a"]),
            self.floats_fn: np.array([15., -10., -1.])
        })
        actual_dc = self._run_sorter(fn=self.strings_fn, order='descending')
        self.assertTrue(expected_dc == actual_dc)

    def test_sorting_by_string_ascending(self):
        expected_dc = DataChunk(**{
            self.ints_fn: np.array([10, 0, 123]),
            self.strings_fn: np.array(["a", "c", "d"]),
            self.floats_fn: np.array([-1., -10, 15.])
        })
        actual_dc = self._run_sorter(fn=self.strings_fn, order='ascending')
        self.assertTrue(expected_dc == actual_dc)

    def _get_dc(self):
        return DataChunk(**{
            self.ints_fn: np.array([10, 0, 123]),
            self.strings_fn: np.array(["a", "c", "d"]),
            self.floats_fn: np.array([-1., -10., 15.])
        })

    def _run_sorter(self, fn, order):
        dc = self._get_dc()
        sorter = ChunkSorter(field_name=fn, order=order)
        dc = sorter(dc)
        return dc


if __name__ == '__main__':
    unittest.main()
