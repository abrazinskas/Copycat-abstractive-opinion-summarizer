import unittest
import numpy as np
from mltoolkit.mldp.tests.common import read_data_from_csv_file
from mltoolkit.mldp.steps.transformers.nlp import Padder
from itertools import product
from mltoolkit.mldp.utils.tools import DataChunk
import copy


class TestPadder(unittest.TestCase):

    def test_2D_padding(self):
        """
        Testing if padding works correctly for common scenarios of 2D data
        (batch_size x sequences).
        Specifically testing whether it produces proper padded sequences, and
        their masks. Also, testing if when symbol_to_mask is provided if it
        correctly masks those symbols.
        """
        field_names = ["text"]
        mask_field_names = ['text_mask']
        data_path = "mldp/tests/data/news.csv"
        pad_symbol = "<PAD>"
        mask_field_name_suffix = "mask"
        padding_modes = ['left', 'right', 'both']
        symbols_to_mask = ["The", "a", "to", "as"]
        axis = 1

        data_chunk = read_data_from_csv_file(data_path, sep="\t")

        # tokenize field values
        for fn in field_names:
            data_chunk[fn] = np.array([seq.split() for seq in data_chunk[fn]])

        for padding_mode, symbol_to_mask in product(padding_modes,
                                                    symbols_to_mask):
            padder = Padder(field_names, pad_symbol=pad_symbol,
                            new_mask_fname=mask_field_names,
                            padding_mode=padding_mode, axis=axis,
                            symbol_to_mask=symbol_to_mask)
            padded_data_chunk = padder(copy.deepcopy(data_chunk))

            for fn, mask_fn in zip(field_names, mask_field_names):
                padded_fv = padded_data_chunk[fn]
                mask = padded_data_chunk[mask_fn]
                original_fv = data_chunk[fn]

                self.assertTrue(len(padded_fv.shape) == 2)
                self._test_padded_values(original_field_values=original_fv,
                                         padded_field_values=padded_fv,
                                         mask=mask, pad_symbol=pad_symbol,
                                         symbol_to_mask=symbol_to_mask)

    def test_3D_padding(self):
        """Light version test to check if the padder works for 3D data."""
        field_name = "dummy"
        mask_field_name = 'dummy_mask'
        pad_symbol = -99
        mask_fn_suffix = "mask"
        padding_mode = "both"
        axis = 2

        data_chunk = DataChunk(**{field_name: np.array([
            [[0, 1, 2], [3, 4, 5], [], [6]],
            [[1], [1, 2], []]
        ])})
        padder = Padder(field_name, pad_symbol=pad_symbol, axis=axis,
                        new_mask_fname=mask_field_name,
                        padding_mode=padding_mode)
        padded_data_chunk = padder(copy.deepcopy(data_chunk))

        original_fv = data_chunk[field_name]
        padded_fv = padded_data_chunk[field_name]
        mask = padded_data_chunk[mask_field_name]

        for ofv, pfv, m in zip(original_fv, padded_fv, mask):
            self._test_padded_values(original_field_values=ofv,
                                     padded_field_values=pfv, mask=m,
                                     pad_symbol=pad_symbol)

    def _test_padded_values(self, original_field_values, padded_field_values,
                            mask, pad_symbol, symbol_to_mask=None):
        self.assertTrue(len(padded_field_values) == len(original_field_values))

        # testing both padded sequences and the produced mask
        for seq_act, seq_exp, m in zip(padded_field_values,
                                       original_field_values,
                                       mask):
            indx = -1
            for c_indx, elem in enumerate(seq_act):
                if elem != pad_symbol:
                    indx += 1
                if elem != pad_symbol and elem != symbol_to_mask:
                    self.assertEqual(elem, seq_exp[indx])
                else:
                    self.assertEqual(0., m[c_indx])


if __name__ == '__main__':
    unittest.main()
