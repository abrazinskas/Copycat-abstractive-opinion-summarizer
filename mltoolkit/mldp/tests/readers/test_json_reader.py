import unittest
from mltoolkit.mldp.steps.readers import JsonReader
from mltoolkit.mldp.steps.transformers.general import ChunkSorter
from mltoolkit.mldp.tests.common import read_data_from_csv_file, \
    concat_data_chunks


class TestJsonReader(unittest.TestCase):

    def test_reading_from_tree_json(self):
        csv_data_path = 'mldp/tests/data/mock_data.csv'
        json_data_path = 'mldp/tests/data/json_reading/mock_data_as_tree.json'
        sorter = ChunkSorter('id', order='ascending')

        exp_dataset = read_csv_data(csv_data_path)
        exp_dataset = sorter(exp_dataset)

        act_dataset = []
        reader = JsonReader()
        for chunk in reader.iter(data_path=json_data_path):
            act_dataset.append(chunk)

        act_dataset = concat_data_chunks(*act_dataset)
        act_dataset = sorter(act_dataset)

        self.assertTrue(exp_dataset == act_dataset)

    def test_reading_from_list_json(self):
        csv_data_path = 'mldp/tests/data/mock_data.csv'
        json_data_path = 'mldp/tests/data/json_reading/mock_data_as_dict.json'
        sorter = ChunkSorter('id', order='ascending')

        exp_dataset = read_csv_data(csv_data_path)
        exp_dataset = sorter(exp_dataset)

        act_data_chunks = []
        reader = JsonReader()
        for chunk in reader.iter(data_path=json_data_path):
            act_data_chunks.append(chunk)

        act_dataset = concat_data_chunks(*act_data_chunks)
        act_dataset = sorter(act_dataset)

        self.assertTrue(exp_dataset == act_dataset)


def read_csv_data(path):
    dataset = read_data_from_csv_file(path, sep=',')
    for k, v in dataset.items():
        if k != 'id':
            dataset[k] = v.astype('str')
    return dataset


if __name__ == '__main__':
    unittest.main()
