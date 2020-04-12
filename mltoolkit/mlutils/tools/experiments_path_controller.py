import os
from mltoolkit.mlutils.helpers.paths_and_files import safe_mkdir, \
    get_immediate_subdirs
from datetime import datetime


class ExperimentsPathController(object):
    """
    This class responsible for creation of paths and necessary folders that are
    used for hosting experiment artifacts, such as event logs, saved models.

    Operates only on local host machine.

    Assumes that a parent subdirectory can contain other folders not related to
    the actual experiments, and those are ignored.

    Folder names are {NAME}{SEP}{SUFFIX} where NAME can be an integer or a date.
    In the latter case, if the same folder exists, then the folder name gets
    suffixed with an integer.
    """

    def __init__(self, folder_name_type='int', sep='__',
                 date_format='%m-%d:%H-%M-%S'):
        if folder_name_type not in ['int', 'date']:
            raise ValueError("The parameter 'folder_name_type' must be 'ints'"
                             " or 'dates'.")
        self.folder_name_type = folder_name_type
        self.sep = sep
        self.date_format = date_format

    def __call__(self, parent_folder_path, suffix=None):
        """
        :param parent_folder_path: the path to an existing or non-existing
                                   parent folder that should host experiment
                                    folders.
        :param suffix: TODO!
        :return: a new path where artifacts can be stored.
        """
        parent_folder_path = os.path.join(os.getcwd(), parent_folder_path)
        safe_mkdir(parent_folder_path)

        # 1. create a new folder's name
        new_folder_name = self._create_new_folder_name(parent_folder_path,
                                                       suffix)

        # 2. create a new folder
        new_folder_path = os.path.join(parent_folder_path, new_folder_name)
        os.mkdir(new_folder_path)

        return new_folder_path

    def _get_last_int_folder_name(self, parent_folder_path):
        """Returns the last folder's name int in the parsed format."""
        sub_dir_names = get_immediate_subdirs(parent_folder_path)
        valid_dir_names = []
        for dn in sub_dir_names:
            try:
                valid_dir_names.append(self._parse_name(dn))
            except StandardError:
                pass
        if valid_dir_names:
            lfn = sorted(valid_dir_names, reverse=True)[0]
        else:
            lfn = None
        return lfn

    def _create_new_folder_name(self, par_dir_path, suffix=None):
        if self.folder_name_type == 'int':
            last_fn_int = self._get_last_int_folder_name(par_dir_path)
            last_fn_int = last_fn_int if last_fn_int else 0
            new_fn = str(last_fn_int + 1)
            new_fn = self.sep.join([new_fn, suffix]) if suffix else new_fn

        elif self.folder_name_type == 'date':
            new_fn = datetime.now().strftime(self.date_format)
            new_fn = self.sep.join([new_fn, suffix]) if suffix else new_fn
            ext_suff = self._create_suffix_if_date_dir_exists(par_dir_path,
                                                              new_fn)
            new_fn = self.sep.join([new_fn, str(ext_suff)]) if ext_suff else \
                new_fn
        else:
            raise ValueError("'folder_name_type' is invalid.")

        return new_fn

    def _parse_name(self, dir_name):
        try:
            parts = dir_name.split(self.sep)
            name = parts[0]
            if self.folder_name_type == 'int':
                return int(name)
            # do not convert to date object because some parts might be missing
            if self.folder_name_type == 'date':
                return name
        except StandardError:
            raise RuntimeError("Could not parse the dir_name '%s'." % dir_name)

    def _create_suffix_if_date_dir_exists(self, base_path, dir_name,
                                          suffix_int=None):
        """Return an int suffix for the end of dir_name if the dir exists."""
        cur_dir_name = self.sep.join([dir_name, str(suffix_int)]) if \
            suffix_int else dir_name
        full_path = os.path.join(base_path, cur_dir_name)
        if os.path.isdir(full_path):
            suffix_int = suffix_int + 1 if suffix_int else 1
            return self._create_suffix_if_date_dir_exists(base_path, dir_name,
                                                          suffix_int=suffix_int)
        else:
            return suffix_int
