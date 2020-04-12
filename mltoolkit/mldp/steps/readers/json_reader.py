from mltoolkit.mldp.steps.readers import BaseReader
from mltoolkit.mlutils.helpers.general import listify
from mltoolkit.mldp.utils.helpers.validation import validate_data_paths
from mltoolkit.mldp.utils.tools import DataChunk
from mltoolkit.mldp.utils.constants.dp import GROUPING_FNAMES
import json
import numpy as np
from collections import OrderedDict
from mltoolkit.mldp.steps.readers.helpers import create_openers_of_valid_files


class JsonReader(BaseReader):
    """
    Single threaded JSON reader for local or s3 (not tested) stored data.
    Yields data-chunks associated with each file.
    """

    def __init__(self, encoding='utf-8', **kwargs):
        """
        :param encoding: encoding of the input json file(s).
        """
        if 'chunk_size' in kwargs:
            raise ValueError("The reader will read whole files at once, so "
                             "please don't provide the 'chunk_size' parameter.")
        super(JsonReader, self).__init__(chunk_size=None, **kwargs)
        self._to_data_chunk = JsonParseToChunk()
        self.encoding = encoding

    def _iter(self, data_path):
        """
        :param data_path: a string corresponding to a location of data
                          (file or folder). If list is provided, will assume
                          multiple data paths.
        """
        try:
            validate_data_paths(data_path)
        except Exception as e:
            raise e

        data_paths = listify(data_path)
        file_openers = create_openers_of_valid_files(data_paths, ext='.json')
        if not file_openers:
            raise ValueError(
                "No valid files to open, please check the provided"
                " 'data_path' (%s). Note that files without the '%s' extension"
                " are ignored." % (data_paths, 'json'))

        for fo in file_openers:
            f = fo(encoding=self.encoding)
            json_dict = json.load(f, object_pairs_hook=OrderedDict)
            data_chunk = self._to_data_chunk(json_dict)
            yield data_chunk
            f.close()


class JsonParseToChunk(object):

    def __call__(self, parsed_json):
        """
        Converts parsed json to data-chunks and returns a data-chunk. Assumes
        that all keys of the json are consistent across instances.

        :param parsed_json: either list or dict format parsed json.
        """
        if isinstance(parsed_json, list):
            return self._data_chunk_from_dicts_list(parsed_json)
        elif isinstance(parsed_json, dict):

            if GROUPING_FNAMES not in parsed_json:
                raise ValueError("The json file is invalid as it has no '%s'"
                                 " meta-data." % GROUPING_FNAMES)
            grouping_fnames = parsed_json[GROUPING_FNAMES]
            del parsed_json[GROUPING_FNAMES]

            return self._data_chunk_from_dicts_tree(parsed_json,
                                                    grouping_fnames)

    @staticmethod
    def _data_chunk_from_dicts_tree(dicts, tree_grouping_fnames):
        """Creates a data-chunk from a tree of data-units (dicts)."""

        def yield_paths_and_leaves(tree, path=None):
            def is_leaf(dct):
                for v in dct.values():
                    if not isinstance(v, list):
                        return False
                return True

            if is_leaf(tree):
                yield path, tree
            else:
                for k in tree.keys():
                    curr_path = [p for p in path] if path else []
                    curr_path.append(k)
                    for r in yield_paths_and_leaves(tree[k], curr_path):
                        yield r

        if not tree_grouping_fnames:
            raise ValueError("Please provide 'tree_grouping_fnames' to parse "
                             "input json files.")
        data_chunk = DataChunk()
        for fn in tree_grouping_fnames:
            data_chunk[fn] = []

        for path, leaf in yield_paths_and_leaves(dicts):
            leaf_size = _get_leaf_size(leaf)
            if len(path) != len(tree_grouping_fnames):
                raise ValueError("Please provide all grouping fields.")

            # storing path values
            for p_val, fn in zip(path, tree_grouping_fnames):
                data_chunk[fn] += [p_val] * leaf_size

            # storing leaf values
            for k, vals in leaf.items():
                assert (isinstance(vals, list))
                if k not in data_chunk:
                    data_chunk[k] = []
                data_chunk[k] += vals

        for k, v in data_chunk.items():
            data_chunk[k] = np.array(v)

        return data_chunk

    @staticmethod
    def _data_chunk_from_dicts_list(list_of_dicts):
        """Creates a data-chunk from list of data-units (dicts)."""
        data_chunk = DataChunk()
        flag = False
        for du in list_of_dicts:
            if not flag:
                for k in du.keys():
                    data_chunk[k] = []
                flag = True
            for k, v in du.items():
                data_chunk[k].append(v)
        for k, v in data_chunk.items():
            data_chunk[k] = np.array(v)
        return data_chunk


def _get_leaf_size(leaf):
    if not _all_equal_list_elems([len(v) for v in leaf.values()]):
        raise ValueError("All leaf field values should be of the same"
                         " size.")
    return len(leaf[list(leaf.keys())[0]])


def _all_equal_list_elems(lst):
    return len(set(lst)) == 1
