import unittest
from mltoolkit.mldp.utils.tools import Vocabulary
from mltoolkit.mldp.utils.constants.vocabulary import UNK
from mltoolkit.mldp.steps.readers import CsvReader
from mltoolkit.mlutils.helpers.paths_and_files import get_file_paths
from mltoolkit.mldp.tests.common import read_data_from_csv_file
import numpy as np
import os


class TestVocabulary(unittest.TestCase):

    def setUp(self):
        self.reader = CsvReader(sep=',')

    def test_creation(self):
        data_path = 'mldp/tests/data/small_chunks/'
        data_source = {"data_path": data_path}
        vocab = Vocabulary(self.reader)
        vocab.create(data_source, "first_name")

        data = read_data_from_csv_file(get_file_paths(data_path))
        unique_first_names = np.unique(data['first_name'])

        for ufn in unique_first_names:
            self.assertTrue(ufn in vocab)

    def test_loading(self):
        tmp_f_path = "dummy_vocab.txt"
        sep = '\t'
        entries = [("first", "1"), ("two", "2"), ("three", "3"), ("four", "4"),
                   ("five", "5"), ("seven", "7")]
        with open(tmp_f_path, 'w') as f:
            for entry in entries:
                f.write(sep.join(entry) + "\n")

        vocab = Vocabulary()
        vocab.load(tmp_f_path, sep="\t")

        for token, count in entries:
            self.assertTrue(token in vocab)
            self.assertTrue(vocab[token].count == int(count))
        os.remove(tmp_f_path)

    def test_when_unk_symbol_is_present(self):
        data_path = 'mldp/tests/data/small_chunks/'
        data_source = {"data_path": data_path}
        vocab = Vocabulary(self.reader)
        vocab.create(data_source, "first_name",
                     add_default_special_symbols=True)

        unk_token = vocab["dummy_token"].token
        self.assertTrue(unk_token == UNK)

    def test_when_unk_symbol_is_absent(self):
        data_path = 'mldp/tests/data/small_chunks/'
        data_source = {"data_path": data_path}
        vocab = Vocabulary(self.reader)
        vocab.create(data_source, "first_name",
                     add_default_special_symbols=False)

        with self.assertRaises(Exception):
            a = vocab["dummy_token"]


if __name__ == '__main__':
    unittest.main()
