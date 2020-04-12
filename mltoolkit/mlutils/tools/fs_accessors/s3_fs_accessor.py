from mltoolkit.mlutils.tools.fs_accessors import BaseFsAccessor
from mltoolkit.mlutils.helpers.paths_and_files import is_s3_path
from mltoolkit.mlutils.helpers.aws import aws_s3_ls
import re
import os
import s3fs


class S3FsAccessor(BaseFsAccessor):
    """
    Specific to the remote AmazonFields S3 file system.
    """

    def __init__(self):
        self.s3_fs = s3fs.S3FileSystem(anon=False)

    def remove_folder_recursively(self, path):
        self.s3_fs.rm(path, True)

    def remove_file(self, path):
        self.s3_fs.rm(path)

    def make_folder(self, path):
        path = self.correct_dir_path(path)
        self.s3_fs.mkdir(path)

    def open_file(self, path, mode='r'):
        return self.s3_fs.open(path, mode)

    def list_dirs(self, path):
        if self.is_file(path):
            raise ValueError("Please provide a valid directory path.")
        path = self.correct_dir_path(path)
        r = []
        for dir_path in self.s3_fs.ls(path):
            if not self.is_file(dir_path):
                r.append(os.path.basename(dir_path))
        return r

    def list_file_paths(self, path):
        if self.is_file(path):
            raise ValueError("Please provide a valid directory path.")
        path = self.correct_dir_path(path)
        r = []
        for dir_path in self.s3_fs.ls(path):
            if self.is_file(dir_path):
                r.append(os.path.basename(dir_path))
        return r

    def path_exists(self, path):
        try:
            aws_s3_ls(path)
        except:
            return False
        return True

    def is_file(self, path):
        return re.match(r'^.*\w+\.\w*$', path)

    def safe_make_folder(self, path):
        if not self.path_exists(path):
            self.make_folder(path)

    def is_valid_path(self, path):
        return is_s3_path(path)

    @staticmethod
    def correct_dir_path(path):
        # incorrectly formatted paths can produce errors, so correct it up-front
        return path if path[-1] == "/" else path + "/"
