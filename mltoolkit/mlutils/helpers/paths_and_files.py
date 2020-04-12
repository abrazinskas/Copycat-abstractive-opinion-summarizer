import os
import re
import errno
import shutil


def get_file_paths(dir_path):
    """
    :param dir_path: self-explanatory.
    :return: a list of file paths that are in the folder. If dir_path is
             actually a file_path it will be returned in the list.
    """
    if not os.path.exists(dir_path):
        raise ValueError("The path '%s' does not exist!" % dir_path)
    if os.path.isdir(dir_path):
        paths = []
        for f_name in os.listdir(dir_path):
            if f_name == '.DS_Store':
                continue
            f_path = os.path.join(dir_path, f_name)
            if os.path.isfile(f_path):
                paths.append(f_path)
    else:
        paths = [dir_path]
    return paths


def iter_file_paths(dir_path):
    """Same as `get_file_paths` but returns an iterator."""
    if not os.path.exists(dir_path):
        raise ValueError("The path '%s' does not exist!" % dir_path)
    if os.path.isdir(dir_path):
        for f_name in os.listdir(dir_path):
            if f_name == '.DS_Store':
                continue
            f_path = os.path.join(dir_path, f_name)
            if os.path.isfile(f_path):
                yield f_path
    else:
        yield dir_path


def get_file_name(file_path):
    fn = os.path.basename(file_path)
    fn = fn.split(".")[0]
    return fn


def safe_mkdir(path):
    if not os.path.exists(path):
        os.makedirs(path)


def is_s3_path(path):
    if re.match(r'^s3:\/\/.*$', path):
        return True
    else:
        return False


def is_file_path(path):
    return re.match(r'^/?([^/ ]+/)*/?\w+\.\w+$', path)


def filter_file_paths_by_extension(file_paths, ext='csv'):
    """
    Filters out file paths that do not have an appropriate extension.

    :param file_paths: list of file path strings
    :param ext: valid extension
    """
    valid_file_paths = []
    for file_path in file_paths:
        ext = ".%s" % ext if ext[0] != '.' else ext
        if not file_path.endswith(ext):
            continue
        valid_file_paths.append(file_path)
    return valid_file_paths


def safe_mkfdir(file_path):
    """Creates folders associated with host of the file."""
    if os.path.dirname(file_path) and not \
            os.path.exists(os.path.dirname(file_path)):
        try:
            os.makedirs(os.path.dirname(file_path))
        except OSError as exc:  # Guard against race condition
            if exc.errno != errno.EEXIST:
                raise


def get_immediate_subdirs(a_dir):
    """Returns a list of subdirectory names."""
    return [name for name in os.listdir(a_dir)
            if os.path.isdir(os.path.join(a_dir, name))]


def get_subdirs_number(path):
    """
    Checks the number of subdirectories, and returns it. Useful for automatic
    output folders generation.
    """
    if not os.path.exists(path):
        return 0
    subdirectories = get_immediate_subdirs(path)
    return len(subdirectories)


def comb_paths(*args):
    return os.path.join(*args)


def remove_dir(folder_path):
    shutil.rmtree(folder_path)
