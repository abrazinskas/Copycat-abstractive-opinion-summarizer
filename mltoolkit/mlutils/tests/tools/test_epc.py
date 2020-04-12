import unittest
import os
from mltoolkit.mlutils.tools import ExperimentsPathController
from shutil import rmtree
import time
from datetime import datetime


class TestEPC(unittest.TestCase):
    """Tests ExperimentsPathController class."""

    def setUp(self):
        self.base_path = 'tmp'
        try:
            os.mkdir(self.base_path)
        except OSError:
            rmtree(self.base_path)
            os.mkdir(self.base_path)

    def tearDown(self):
        rmtree(self.base_path)

    def test_int_first_dir(self):
        """Testing if can create first directory with a proper int number."""
        path = os.path.join(self.base_path, "ints0")
        exp_new_path = os.path.join(path, '1')
        epc = ExperimentsPathController(folder_name_type='int')
        new_path = epc(path)
        self.assertTrue(exp_new_path, new_path)

    def test_int_suffix(self):
        """
        1. Testing whether it creates a proper custom suffix for dir names.
        2. Whether a newly initialized epc parsed dir names properly.
        """
        path = os.path.join(self.base_path, "ints1")
        suffix = "dummy"
        sep = "^"

        # 1.
        n = 4
        epc = ExperimentsPathController(folder_name_type='int', sep=sep)
        exp_dirs = [sep.join([str(i), suffix]) for i in range(1, n + 1)]

        for _ in range(n):
            epc(path, suffix=suffix)

        act_dirs = os.listdir(path)

        self.assertTrue(set(act_dirs) == set(exp_dirs))

        # 2.
        m = 6
        epc = ExperimentsPathController(folder_name_type='int', sep=sep)
        new_exp_dirs = [str(i) for i in range(n + 1, n + m + 1)]
        for _ in range(m):
            epc(path)

        act_dirs = os.listdir(path)

        self.assertTrue(set(act_dirs) == set(exp_dirs + new_exp_dirs))

    def test_date_dir_names(self):
        """Testing correctness of date type produced dir_names."""
        sep = "__"
        n = 5
        date_format = '%m-%d:%H-%M-%S'
        path = os.path.join(self.base_path, "date")

        epc = ExperimentsPathController(folder_name_type='date', sep=sep,
                                        date_format=date_format)

        for _ in range(n):
            epc(path)
            exp_dir_name = datetime.now().strftime(date_format)
            dirs = os.listdir(path)
            self.assertTrue(exp_dir_name in dirs)
            time.sleep(1)

    def test_date_dir_names_duplicates(self):
        """
        Testing if a correct extra prefix is produced when multiple runs happen
        in the same second.
        """
        sep = "__"
        n = 5
        date_format = '%m-%d:%H-%M-%S'
        path = os.path.join(self.base_path, "date")

        epc = ExperimentsPathController(folder_name_type='date', sep=sep,
                                        date_format=date_format)

        for _ in range(n):
            epc(path)

        exp_dir_name = datetime.now().strftime(date_format)
        exp_dirs = [sep.join([exp_dir_name, str(i)]) for i in range(1, n)] + \
                   [exp_dir_name]
        act_dirs = os.listdir(path)

        self.assertTrue(set(act_dirs) == set(exp_dirs))


if __name__ == '__main__':
    unittest.main()
