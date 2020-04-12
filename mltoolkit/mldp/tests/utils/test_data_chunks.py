import unittest
from mltoolkit.mldp.tests.common import generate_data_chunk
from mltoolkit.mldp.utils.errors import DataChunkError
from mltoolkit.mldp.utils.tools.dc_writers import JsonWriter
from itertools import product
from mltoolkit.mldp.utils.tools import DataChunk
from mltoolkit.mlutils.helpers.paths_and_files import safe_mkdir, remove_dir
import os
import numpy as np
import json
from random import seed
from copy import deepcopy

seed(42)


class TestDataChunks(unittest.TestCase):
    """Tests different aspects of the data-chunk class."""

    def setUp(self):
        self.tmp_folder = ".tmp"
        safe_mkdir(self.tmp_folder)

    def tearDown(self):
        if os.path.exists(self.tmp_folder):
            remove_dir(self.tmp_folder)

    def test_valid_chunks(self):
        chunk_sizes = [1, 20, 12, 1023, 100]
        attrs_numbers = [1, 3, 10, 25, 9]

        for attrs_number, chunk_size in product(attrs_numbers, chunk_sizes):
            good_chunk = generate_data_chunk(attrs_number, chunk_size)
            try:
                good_chunk.validate()
            except Exception:
                raise self.assertTrue(False)

    def test_chunks_with_wrong_value_types_in_constr(self):
        """Testing if an error is thrown for invalid chunk value types"""
        chunk_size = 100
        attrs_numbers = [2, 3, 10]
        invalid_values = ["dummy_val", [1231, 123123, 12], (), object, 1.23]

        for attrs_number, invalid_val in product(attrs_numbers, invalid_values):
            chunk = generate_data_chunk(attrs_number, chunk_size)
            attr_to_alter = np.random.choice(list(chunk.keys()), 1)[0]
            with self.assertRaises(DataChunkError):
                chunk[attr_to_alter] = invalid_val
                chunk.validate()

    def test_chunks_with_different_value_array_sizes(self):
        chunk_size = 100
        attrs_numbers = [2, 3, 10, 25, 9]

        for attrs_number in attrs_numbers:
            chunk = generate_data_chunk(attrs_number, chunk_size)
            attr_to_alter = np.random.choice(list(chunk.keys()), 1)[0]
            chunk[attr_to_alter] = chunk[attr_to_alter][:-1]
            with self.assertRaises(DataChunkError):
                chunk.validate()

    def test_json_writing_tree(self):
        """Testing whether tree-like json dumping is working correctly."""
        expected_output_fp = "mldp/tests/data/chunk_writing/expected_tree_dump.json"
        with open(expected_output_fp) as f:
            expected_output = json.load(f)
        output_fp = os.path.join(self.tmp_folder, "tree_dump.json")

        dc = self._get_dummy_dc()

        writer = JsonWriter(f=open(output_fp, 'w'),
                            grouping_fnames=["country", "shop_id"])
        with writer as w:
            w.write(dc)

        with open(output_fp) as f:
            actual_output = json.load(f)

        # sorting trees
        actual_output = ordered(actual_output)
        expected_output = ordered(expected_output)

        self.assertTrue(actual_output == expected_output)

    def test_json_writing(self):
        """Testing whether list of dicts json dumping is working correctly."""
        expected_output_fp = "mldp/tests/data/chunk_writing/expected_dump.json"
        with open(expected_output_fp) as f:
            expected_output = json.load(f)
        output_fp = os.path.join(self.tmp_folder, "_dump.json")

        dc = self._get_dummy_dc()

        with JsonWriter(f=open(output_fp, 'w')) as w:
            w.write(dc)

        with open(output_fp) as f:
            actual_output = json.load(f)

        # sorting trees
        actual_output = ordered(actual_output)
        expected_output = ordered(expected_output)

        self.assertTrue(actual_output == expected_output)

    def test_field_values_access(self):

        arrays_size = 40
        names = ["one", "two", "three", "four"]

        for _ in range(10):
            data = {name: np.random.rand(arrays_size, 1) for name in names}
            data_chunk = DataChunk(**deepcopy(data))
            for name in names:
                self.assertTrue((data_chunk[name] == data[name]).all())

    def test_specific_fvalues_access(self):
        arrays_size = 40
        names = ["one", "two", "three", "four"]

        for _ in range(10):
            data = {name: np.random.rand(arrays_size) for name in names}
            data_chunk = DataChunk(**deepcopy(data))

            for r_name in np.random.choice(names, size=10, replace=True):
                for r_indx in np.random.randint(0, 40, size=100):
                    res = (data_chunk[r_indx, r_name] == data[r_name][r_indx])
                    self.assertTrue(res)

    def test_data_units_inval_access(self):
        """When data-chunk is incorrect, it should throw an error."""

        dc = DataChunk(test=[1, 22, 3, 4, 5], dummy=[99, 2, 3])

        with self.assertRaises(DataChunkError):
            du = dc[0]

        dc["test"] = np.array(dc["test"])
        dc['dummy'] = np.array(dc['dummy'])

        with self.assertRaises(DataChunkError):
            du = dc[0]

        dc['dummy'] = np.append(dc['dummy'], 0)
        dc['dummy'] = np.append(dc['dummy'], 1)

        du = dc[0]

        self.assertTrue(du['test'] == 1)
        self.assertTrue(du['dummy'] == 99)

        du = dc[1]

        self.assertTrue(du['test'] == 22)
        self.assertTrue(du['dummy'] == 2)

    def test_valid_data_units_deletion(self):
        dc = DataChunk(one=np.array([1, 2, 3, 4]),
                       two=np.array([0, 10, 11, 24]))

        del dc[2]

        self.assertTrue((np.array([1, 2, 4]) == dc['one']).all())
        self.assertTrue((np.array([0, 10, 24]) == dc['two']).all())

    def test_invalid_data_units_deletion(self):
        """Deletion by data-unit index from data-chunks should not work."""
        dc = DataChunk(one=[1, 2, 3, 4], two=[10, 20, 30, 40, 50, 60])

        self.assertFalse(dc.valid)

        try:
            del dc[2]
        except ValueError as e:
            self.assertTrue(True)
        else:
            self.assertTrue(False)

    def test_appending_data_units_to_valid_dc(self):

        act_dc = DataChunk(test=np.array([], dtype='int64'),
                           dummy=np.array([], dtype='int64'))

        act_dc.append({"test": 1, "dummy": 2})
        act_dc.append({"test": 3, "dummy": 4})

        exp_dc = DataChunk(test=np.array([1, 3]), dummy=np.array([2, 4]))

        self.assertTrue(act_dc == exp_dc)

    def test_appending_data_units_to_invalid_dc(self):

        act_dc = DataChunk(test=np.array([], dtype='int64'),
                           dummy=np.array([1], dtype='int64'))

        with self.assertRaises(DataChunkError):
            act_dc.append({"test": 1, "dummy": 2})

    def test_setting_data_units(self):
        act_dc = DataChunk(test=np.array([1, 2, 3, 4, 5], dtype='int64'),
                           dummy=np.array([6, 7, 8, 9, 10], dtype='int64'))
        act_dc[0] = {"test": 0, "dummy": 0}
        act_dc[3] = {"test": 20, "dummy": 30}

        exp_dc = DataChunk(test=np.array([0, 2, 3, 20, 5]),
                           dummy=np.array([0, 7, 8, 30, 10]))

        self.assertTrue(act_dc == exp_dc)

    def test_modification_of_data_units(self):
        """Selecting specific data-units and altering their values."""
        act_dc = DataChunk(test=np.array([1, 2, 3, 4]),
                           dummy=np.array([11., 12., 13., 14.]))
        act_du1 = act_dc[0]
        act_du1['test'] += 100

        act_du2 = act_dc[3]
        act_du2['dummy'] += 5

        exp_dc = DataChunk(test=np.array([101, 2, 3, 4]),
                           dummy=np.array([11., 12., 13., 19.]))

        self.assertTrue(act_dc == exp_dc)

    @staticmethod
    def _get_dummy_dc():
        dc = DataChunk()
        dc["country"] = np.array(["UK", "UK", "UK", "DK", "DK"])
        dc["shop_id"] = np.array(['1', '1', '1', '2', '3'])
        dc["product_id"] = np.array([11, 12, 13, 101, 101])
        dc["sales"] = np.array([0, 1, 2, 5, 6])
        return dc


def ordered(obj):
    if isinstance(obj, dict):
        return sorted((k, ordered(v)) for k, v in obj.items())
    if isinstance(obj, list):
        return sorted(ordered(x) for x in obj)
    else:
        return obj


if __name__ == '__main__':
    unittest.main()
