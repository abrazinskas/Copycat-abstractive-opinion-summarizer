import unittest
import torch as T
import numpy as np
from copycat.utils.helpers.modelling import group_att_over_input, \
    optimize_att_tens
from copycat.utils.helpers.data import get_all_but_one_rev_indxs, \
    create_optimized_selector


class TestModellingFuncs(unittest.TestCase):

    def test_get_all_but_one_rev_indxs(self):
        group2rev_indxs = np.array([[0, 1, 2, 4], [5, 0, 0, 0], [6, 3, 0, 0]])
        group2rev_mask = np.array(
            [[1., 1., 1., 1.], [1., 0., 0, 0], [1, 1, 0, 0]])
        exp_other_rev_indxs = np.array([[0, 1, 2, 4], [0, 1, 2, 4],
                                        [0, 1, 2, 4], [6, 3, 0, 0],
                                        [0, 1, 2, 4], [5, 0, 0, 0],
                                        [6, 3, 0, 0]])
        exp_other_rev_mask = np.array([[0., 1., 1., 1.], [1., 0., 1., 1.],
                                       [1., 1., 0., 1.], [1., 0., 0., 0.],
                                       [1., 1., 1., 0.], [0., 0., 0., 0.],
                                       [0., 1., 0., 0.]])

        act_other_rev_indxs, \
        act_other_rev_mask = get_all_but_one_rev_indxs(group2rev_indxs,
                                                       group2rev_mask)

        self.assertTrue((act_other_rev_indxs == exp_other_rev_indxs).all())
        self.assertTrue((act_other_rev_mask == exp_other_rev_mask).all())

    def test_get_all_but_one_rev_indxs2(self):
        summ2rev_indxs = np.array([[4, 0, 0], [2, 1, 0], [3, 0, 5]])
        summ2rev_mask = np.array([[1., 0., 0.], [1., 1., 0.], [1., 1., 1.]])
        exp_other_rev_indxs = np.array([[3, 0, 5], [2, 1, 0],
                                        [2, 1, 0], [3, 0, 5],
                                        [4, 0, 0], [3, 0, 5]])
        exp_other_rev_mask = np.array([[1., 0., 1.], [1., 0., 0.],
                                       [0., 1., 0.], [0., 1., 1.],
                                       [0., 0., 0.], [1., 1., 0.]])

        act_other_rev_indxs, \
        act_other_rev_mask = get_all_but_one_rev_indxs(summ2rev_indxs,
                                                       summ2rev_mask)

        self.assertTrue((act_other_rev_indxs == exp_other_rev_indxs).all())
        self.assertTrue((act_other_rev_mask == exp_other_rev_mask).all())

    def test_get_att_over_input(self):
        """Testing concatenation of review hiddens belonging to same group."""
        hiddens = T.tensor([[[0.1, 0.2], [10.1, 11.1]],
                            [[1, 2], [10, 11]],
                            [[3, 4], [30, 31]],
                            [[5, 6], [50, 51]],
                            [[7, 8], [70, 71]]
                            ])

        mask = T.tensor([[1., 1.], [1., 1.], [1., 0.], [1., 1.], [1., 0.]])
        indxs = T.tensor([[3, 0], [1, 2], [4, 0]])
        indxs_mask = T.tensor([[1., 1.], [1., 1.], [1., 0.]])

        exp_att_vals = T.tensor([[[5., 6.], [50, 51], [0.1, 0.2], [10.1, 11.1]],
                                 [[1, 2, ], [10, 11], [3, 4], [30, 31]],
                                 [[7, 8], [70, 71], [0., 0.], [0., 0.]]
                                 ])
        exp_att_vals_mask = T.tensor([[1., 1., 1., 1.],
                                      [1., 1., 1., 0.],
                                      [1., 0., 0., 0.]])
        act_att_vals, \
        act_att_keys, \
        act_att_vals_mask = group_att_over_input(inp_att_vals=hiddens,
                                                 inp_att_keys=hiddens,
                                                 inp_att_mask=mask,
                                                 att_indxs=indxs,
                                                 att_indxs_mask=indxs_mask)

        self.assertTrue((exp_att_vals == act_att_keys).all())
        self.assertTrue((exp_att_vals == act_att_vals).all())
        self.assertTrue((exp_att_vals_mask == act_att_vals_mask).all())

    def test_optimize_att_tensor(self):
        tens = T.tensor(([[1, 1, 0, 0, 2, 2, 2, 0, 3, 3],
                          [1.1, 1.1, 0, 2.2, 2.2, 2.2, 0., 0., 3.3,
                           0.]])).unsqueeze(-1)
        mask = T.tensor([[1, 1, 0, 0, 1, 1, 1, 0, 1, 1],
                         [1, 1, 0, 1, 1, 1, 0, 0, 1, 0]], dtype=T.float32)

        exp_tens = T.tensor(([[1, 1, 2, 2, 2, 3, 3],
                              [1.1, 1.1, 2.2, 2.2, 2.2, 3.3, 0.]])).unsqueeze(
            -1)
        exp_mask = T.tensor([[1, 1, 1, 1, 1, 1, 1],
                             [1, 1, 1, 1, 1, 1, 0]], dtype=T.float32)

        act_tens, act_mask = optimize_att_tens(mask, tens)

        self.assertTrue(T.allclose(exp_tens, act_tens[0]))
        self.assertTrue(T.allclose(exp_mask, act_mask))

    def test_create_optimized_selector(self):
        concat_tens = T.tensor(([[1, 1, 0, 0, 2, 2, 2, 0, 3, 3],
                                 [1.1, 1.1, 0, 2.2, 2.2, 2.2, 0., 0., 3.3,
                                  0.]])).unsqueeze(-1)
        concat_mask = np.array([[1, 1, 0, 0, 1, 1, 1, 0, 1, 1],
                                [1, 1, 0, 1, 1, 1, 0, 0, 1, 0]])
        exp_tens = T.tensor(([[1, 1, 2, 2, 2, 3, 3],
                              [1.1, 1.1, 2.2, 2.2, 2.2, 3.3, 0.]])).unsqueeze(
            -1)
        exp_mask = np.array([[1, 1, 1, 1, 1, 1, 1],
                             [1, 1, 1, 1, 1, 1, 0]])

        exp_indxs = np.array([[0, 1, 4, 5, 6, 8, 9],
                              [0, 1, 3, 4, 5, 8, 0]])

        indxs, indxs_mask = create_optimized_selector(concat_mask)

        self.assertTrue((exp_indxs == indxs).all())

        # creating a more compact version of the previously concat tensor
        act_tens = concat_tens[T.arange(concat_tens.size(0)).unsqueeze(-1),
                               indxs] * T.tensor(indxs_mask[:, :, np.newaxis],
                                                 dtype=T.float32)

        self.assertTrue((act_tens == exp_tens).all())
        self.assertTrue((indxs_mask == exp_mask).all())


if __name__ == '__main__':
    unittest.main()
