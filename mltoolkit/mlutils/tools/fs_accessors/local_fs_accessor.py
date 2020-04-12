from mltoolkit.mlutils.tools.fs_accessors import BaseFsAccessor
from mltoolkit.mlutils.helpers.paths_and_files import safe_mkdir, \
    get_file_paths as gfp
import os
import shutil
import codecs


class LocalFsAccessor(BaseFsAccessor):
    def __init__(self):
        super(LocalFsAccessor, self).__init__()

    def remove_file(self, path):
        os.remove(path)

    def make_folder(self, path):
        os.makedirs(path)

    def open_file(self, path, mode='r', encoding='utf-8'):
        return codecs.open(path, mode, encoding=encoding)

    def list_dirs(self, path):
        # ignore hidden files
        return [dir_name for dir_name in os.listdir(path) if
                not self.is_file(os.path.join(path, dir_name))]

    def list_file_paths(self, path):
        return gfp(path)

    def path_exists(self, path):
        return os.path.exists(path)

    def is_file(self, path):
        return os.path.isfile(path)

    def remove_folder_recursively(self, path):
        shutil.rmtree(path)

    def safe_make_folder(self, path):
        safe_mkdir(path)

    def is_valid_path(self, path):
        pass
