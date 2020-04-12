from mltoolkit.mldp.steps.preprocessors import BasePreProcessor
from mltoolkit.mlutils.helpers.paths_and_files import get_file_paths
from numpy import random


class FileShuffler(BasePreProcessor):
    """Reads the paths of files; assigns them to `data_path` in a shuffled
    order, and passes them along in the data-source.
    """

    def __init__(self, **kwargs):
        super(FileShuffler, self).__init__(**kwargs)

    def __call__(self, data_path, **kwargs):
        file_paths = []
        if not isinstance(data_path, list):
            data_path = [data_path]
        for _data_path in data_path:
            file_paths += get_file_paths(_data_path)

        random.shuffle(file_paths)

        new_data_source = kwargs
        new_data_source['data_path'] = file_paths

        return new_data_source
