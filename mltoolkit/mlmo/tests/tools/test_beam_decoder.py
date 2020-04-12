import unittest
import torch as T
from mltoolkit.mlmo.generation import Beamer
from mltoolkit.mlmo.utils.tools import DecState
import numpy as np


class TestBeamDecoder(unittest.TestCase):
    """
    Note that this test ignores the fact that word scores should be log
    probabilities.
    """

    def test_simple_output(self):
        """Hidden state independent test."""
        beam_size = 2
        max_steps = 3

        vocab = {0: "a", 1: "b", 2: "c", 3: "<pad>", 4: "<s>", 5: "<e>"}
        exp_seqs = [[4, 2, 5], [4, 0, 0, 5]]

        init_hidden = T.tensor([[0., 0., 0.], [0., 0., 0.]], dtype=T.float32)
        init_dec_state = DecState(rec_vals={"hidden": init_hidden})

        dec = Dec()
        beam_decoder = Beamer(decoding_func=dec.dummy_dec_func, start_id=4,
                              beam_size=beam_size, end_id=5,
                              validate_dec_out=False)

        act_seqs, _ = beam_decoder(init_dec_state=init_dec_state,
                                   max_steps=max_steps)
        self.assertTrue((exp_seqs == act_seqs))

    def test_hidden_dependent_output(self):
        beam_size = 2
        max_steps = 3

        vocab = {0: "a", 1: "b", 2: "c", 3: "<pad>", 4: "<s>", 5: "<e>"}
        exp_seqs = [[4, 1, 0, 5]]
        init_hidden = T.tensor([[0.]], dtype=T.float32)

        dec = Dec()
        beam_decoder = Beamer(decoding_func=dec.hidden_dep_dec_func,
                              start_id=4, end_id=5, validate_dec_out=False,
                              n_best=beam_size, beam_size=beam_size)

        init_dec_state = DecState(rec_vals={"hidden": init_hidden})

        act_seqs, _ = beam_decoder(init_dec_state, max_steps=max_steps)

        self.assertTrue((exp_seqs == act_seqs))

    # def test_coll_vals(self):
    #     """
    #     Testing whether the decoder correctly collects additional artifacts
    #     produced by the decoder.
    #     """
    #     beam_size = 2
    #     max_steps = 3
    #     raise NotImplementedError


class Dec:
    def __init__(self):
        self.state = -1

    def dummy_dec_func(self, prev_word_ids, hidden):
        t1 = [
            [1.1, 0., 1., 0., 0., 0.],
            [0., 0., 0., 10., 10., 10.],  # this will be ignored
            [1.1, 1., 0., 0., 0., 0.],
            [0., 0., 0., 10., 0., 0.]  # this will be ignored
        ]

        t2 = [
            [0., 1., 2., 0., 0., 0.],
            [0., 0., 0., 0., 0., 6.],
            [1.1, 0., 0., 0., 0., 0.],
            [0., 1., 0., 0., 0., 0.]
        ]
        t3 = [
            [4., 9999., 3., 10., 133., 5.],
            [0., 0., 0., 0., 0., 1.],
            [0., 0., 0., 0., 0., 1.],
            [0., 0., 0., 0., 0., 1.]
        ]
        self.state += 1

        if self.state == 0:
            word_scores = T.tensor(t1)
        elif self.state == 1:
            word_scores = T.tensor(t2)
        elif self.state == 2:
            word_scores = T.tensor(t3)
        else:
            raise ValueError("The decoding func supports only 3 steps!")

        return DecState(word_scores=word_scores, rec_vals={"hidden": hidden})

    def hidden_dep_dec_func(self, prev_word_ids, hidden):

        t1 = [
            [1., 1.1, 0., 0.0, 0.0, 0.0],
            [10., 0., 50., 0., 0., 0.]  # this will be ignored
        ]

        t2 = [
            [0.2, 0., 0.1, 0., 0., 0.],
            [0., 0., 0., 1.1, 0., 2.]
        ]
        t3 = [
            [4., 99., 331., 10., 133., 53.],  # should be ignored
            [0., 0., 0., -3.0000, 0., 0.001]
        ]

        self.state += 1

        if self.state == 0:
            hidden[0, 0] += 1.
            return DecState(T.tensor(t1), rec_vals={"hidden": hidden})

        if self.state == 1:
            t2 = T.tensor(t2)
            t2[0, :] += hidden[0, 0]
            return DecState(t2, rec_vals={"hidden": T.tensor([[1.2], [2.]])})

        if self.state == 2:
            t3 = T.tensor(t3)
            t3[0, :] += hidden[0, 0]
            t3[1, :] += hidden[1, 0]
            return DecState(t3, rec_vals={"hidden": T.tensor([[0.], [0.]])})

        raise ValueError("The decoding func supports only 3 steps!")

    def val_coll_dec_func(self, prev_word_ids, hidden):
        raise NotImplementedError


if __name__ == '__main__':
    unittest.main()
