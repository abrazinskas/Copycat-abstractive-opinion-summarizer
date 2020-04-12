import unittest
from mltoolkit.mldp.steps.transformers import SeqWrapper
from mltoolkit.mldp.utils.tools import DataChunk
import numpy as np
from copy import deepcopy


class TestSeqWrapper(unittest.TestCase):
    def setUp(self):
        self.start = "<S>"
        self.end = "<E>"
        self.field_name = 'dummy'
        self.wrapper = SeqWrapper(fname=self.field_name,
                                  start_el=self.start, end_el=self.end)

    def test_for_str(self):
        def wrapper(seq):
            return "%s %s %s" % (self.start, seq, self.end)

        seqs = np.array(["one two three four five",
                         "1 2 3 4 5",
                         "I II III"], dtype="object")

        expected_dc, actual_dc = self.run_standard_wrapper(seqs, wrapper)
        self.assertTrue(expected_dc == actual_dc)

    def test_for_unicode(self):
        def wrapper(seq):
            return u"%s %s %s" % (self.start, seq, self.end)

        seqs = np.array([u"one two three four five",
                         u"1 2 3 4 5",
                         u"I II III"], dtype="object")

        expected_dc, actual_dc = self.run_standard_wrapper(seqs, wrapper)
        self.assertTrue(expected_dc == actual_dc)

    def test_for_list(self):
        def wrapper(seq):
            return [self.start] + seq + [self.end]

        seqs = np.zeros((3,), dtype='object')
        seqs[0] = ["one", "two", "three", "four", "five"]
        seqs[1] = ["1", "2", "3", "4", "5"]
        seqs[2] = ["I", "II", "III"]

        expected_dc, actual_dc = self.run_standard_wrapper(seqs, wrapper)
        self.assertTrue(expected_dc == actual_dc)

    def test_for_array(self):
        def wrapper(seq):
            return np.concatenate((np.array([self.start]), seq,
                                   np.array([self.end])))

        seqs = np.array([
            np.array(["one", "two", "three", "four", "five"], dtype="object"),
            np.array(["1", "2", "3", "4", "5"], dtype="object"),
            np.array(["I", "II", "III"], dtype="object")])

        expected_dc, actual_dc = self.run_standard_wrapper(seqs, wrapper)
        self.assertTrue(expected_dc == actual_dc)

    def test_for_array_of_ints(self):
        """Special case where the input is [N, K] array to be wrapped."""
        start_id = 1000
        end_id = 1001

        wrapper = SeqWrapper(fname=self.field_name, start_el=start_id,
                             end_el=end_id)
        seqs = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])

        exp_dc = DataChunk(**{self.field_name:
                                  np.array([[start_id, 1, 2, 3, end_id],
                                            [start_id, 4, 5, 6, end_id],
                                            [start_id, 7, 8, 9, end_id]])
                              })

        act_dc = DataChunk(**{self.field_name: seqs})
        act_dc = wrapper(act_dc)

        self.assertTrue(exp_dc == act_dc)

    def run_standard_wrapper(self, seqs, wrapper_func):
        """A common method among tests"""
        exp_dc = self.get_expected_dc(seqs, wrapper_func)
        act_dc = DataChunk(**{self.field_name: seqs})
        act_dc = self.wrapper(act_dc)
        return exp_dc, act_dc

    def get_expected_dc(self, seqs, wrapper_func):
        exp_dc = DataChunk(**{self.field_name: deepcopy(seqs)})
        for indx in range(len(exp_dc)):
            seq = exp_dc[self.field_name][indx]
            exp_dc[self.field_name][indx] = wrapper_func(seq)
        return exp_dc


if __name__ == '__main__':
    unittest.main()
