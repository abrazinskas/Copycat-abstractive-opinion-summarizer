import unittest
from mltoolkit.mldp.steps.transformers.nlp import SeqLenComputer
from mltoolkit.mldp.utils.tools import DataChunk
from copy import deepcopy
import numpy as np


class TestSeqLenComputer(unittest.TestCase):

    def test_output(self):
        fn = "dummy"
        new_fn = "dummy_len"
        data = [[1, 2, 3], [12], ["a", "b", "d", "e"]]

        actual_dc = DataChunk(**{fn: np.array(deepcopy(data))})
        expected_dc = DataChunk(**{fn: np.array(deepcopy(data)),
                                   new_fn: np.array([3, 1, 4])})
        slc = SeqLenComputer(fname=fn, new_len_fname=new_fn)
        actual_dc = slc(actual_dc)

        self.assertTrue(actual_dc == expected_dc)


if __name__ == '__main__':
    unittest.main()
