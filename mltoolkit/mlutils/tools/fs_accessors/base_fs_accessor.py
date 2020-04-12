class BaseFsAccessor(object):
    """
    Base class for accessing file systems. For example local and cloud(S3) ones.
    Provides the common interface to access different file systems.
    """

    def remove_folder_recursively(self, path):
        raise NotImplementedError

    def remove_file(self, path):
        raise NotImplementedError

    def make_folder(self, path):
        raise NotImplementedError

    def open_file(self, path, mode='r'):
        raise NotImplementedError

    def list_dirs(self, path):
        raise NotImplementedError

    def list_file_paths(self, path):
        raise NotImplementedError

    def path_exists(self, path):
        raise NotImplementedError

    def is_file(self, path):
        raise NotImplementedError

    def safe_make_folder(self, path):
        raise NotImplementedError

    def is_valid_path(self, path):
        """
        Tests if the path has a proper format to be accessed by the object.
        :return: True/False
        """
        raise NotImplementedError
