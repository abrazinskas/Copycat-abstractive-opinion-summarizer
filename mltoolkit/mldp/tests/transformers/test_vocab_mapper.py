import unittest
from mltoolkit.mldp.tests.common import read_data_from_csv_file
from mltoolkit.mldp.utils.tools import Vocabulary, DataChunk
from mltoolkit.mldp.steps.transformers.nlp import VocabMapper
from mltoolkit.mldp.steps.readers import CsvReader
import copy
import numpy as np


class TestVocabMapper(unittest.TestCase):

    def test_vocabulary_mapper(self):
        """Testing whether the mapper allows to map back and forth field values.
        """
        data_path = 'mldp/tests/data/mock_data.csv'
        target_fields = ["first_name", "last_name", "email", "gender"]

        reader = CsvReader(sep=',')
        vocab = Vocabulary(reader)

        for target_field in target_fields:
            vocab.create(data_source={"data_path": data_path},
                         data_fnames=target_field)

            data = read_data_from_csv_file(data_path)
            data_original = copy.deepcopy(data)

            mapper_to = VocabMapper({target_field: vocab}, "id")
            mapper_back = VocabMapper({target_field: vocab}, "token")

            data = mapper_to(data)
            data = mapper_back(data)

            self.assertTrue((data[target_field] == data_original[target_field])
                            .all())

    def test_vocabulary_mapper_multidim_lists(self):
        """Testing whether the mapper can map multi-dim lists."""
        target_field_name = "dummy"
        symbols_attr = "id"

        data_chunk = DataChunk(**{target_field_name: np.array([
            [["one"], ["two"]],
            [["three"], ["four", "five", "six"]]
        ], dtype="object")})
        exp_val = np.empty(2, dtype="object")
        exp_val[0] = np.array([[1], [2]])
        exp_val[1] = np.array([[3], [4, 5, 6]])
        expected_output_chunk = DataChunk(**{target_field_name: exp_val})

        # creating and populating a vocab
        vocab = Vocabulary()
        vocab.add_symbol("zero")
        vocab.add_symbol("one")
        vocab.add_symbol("two")
        vocab.add_symbol("three")
        vocab.add_symbol("four")
        vocab.add_symbol("five")
        vocab.add_symbol("six")

        mapper = VocabMapper({target_field_name: vocab},
                             symbols_attr=symbols_attr)
        actual_output_chunk = mapper(copy.deepcopy(data_chunk))

        self.assertTrue(actual_output_chunk == expected_output_chunk)

    def test_vocabulary_mapper_mixed_field_values(self):
        """Testing whether the mapper can map multi-dim mixed field values."""
        target_field_name = "dummy"
        symbols_attr = "id"

        data_chunk = DataChunk(**{target_field_name: np.array([
            [["one"], np.array(["two", "one"])],
            [["three"], np.array(["four", "five", "six"])]
        ], dtype="object")})
        expected_output_chunk = DataChunk(**{target_field_name: np.array([
            [[1], np.array([2, 1])],
            [[3], np.array([4, 5, 6])]
        ], dtype="object")})

        # creating and populating a vocab
        vocab = Vocabulary()
        vocab.add_symbol("zero")
        vocab.add_symbol("one")
        vocab.add_symbol("two")
        vocab.add_symbol("three")
        vocab.add_symbol("four")
        vocab.add_symbol("five")
        vocab.add_symbol("six")

        mapper = VocabMapper({target_field_name: vocab},
                             symbols_attr=symbols_attr)
        actual_output_chunk = mapper(data_chunk)

        self.assertTrue(actual_output_chunk == expected_output_chunk)


if __name__ == '__main__':
    unittest.main()
