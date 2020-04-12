from mltoolkit.mldp.utils.constants.dp import TERMINATION_TOKEN
from mltoolkit.mlutils.tools.fs_accessors.local_fs_accessor import \
    LocalFsAccessor
from mltoolkit.mlutils.helpers.paths_and_files import \
    filter_file_paths_by_extension
from functools import partial as fun_partial


def populate_queue_with_chunks(f, itr_creator, queue):
    """
    This function is used by thread workers in order to load and store data
    chunks to the common chunk queue.

    :param itr_creator: a function that creates an iterable over data-chunks.
    :param queue: self-explanatory.
    """
    try:
        it = itr_creator(f)
        for data_chunk in it:
            queue.put(data_chunk)
    except Exception as e:
        queue.put(e)
        return
    queue.put(TERMINATION_TOKEN)


def create_openers_of_valid_files(paths, ext='csv', encoding='utf-8'):
    """
    Returns a list of file opener functions, on call return a file object.
    In such a way we hide the details on how to open files of different types
    and avoid opening all files at once.
    """
    valid_file_openers = []
    for path in paths:
        fs = LocalFsAccessor()
        file_paths = fs.list_file_paths(path)
        valid_file_paths = filter_file_paths_by_extension(file_paths, ext=ext)
        valid_file_openers += [fun_partial(fs.open_file, path=p, mode='r',
                                           encoding=encoding)
                               for p in valid_file_paths]
    return valid_file_openers
