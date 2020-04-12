from luigi import Task, Parameter, LocalTarget
from preprocessing.fields import OutputFields
from mltoolkit.mlutils.helpers.paths_and_files import iter_file_paths, \
    comb_paths, \
    safe_mkfdir, get_file_name
from sacremoses import MosesTokenizer
from functools import partial
from preprocessing.helpers.data_utils import read_csv_file, \
    write_group_to_csv, get_act_out_dir_path
import os
from preprocessing.steps import PrepareFile
from preprocessing.global_config import GlobalConfig
from mltoolkit.mlutils.helpers.logging import init_logger
from logging import getLogger
from _csv import Error as CsvError

mt = MosesTokenizer()
tokenizer = partial(mt.tokenize, escape=False)

FOLDER_NAME = "2.tok"
LOGGER_NAME = os.path.basename(__file__)
logger = getLogger(LOGGER_NAME)


class Tokenize(Task):
    """Tokenizes texts in a folder that is prepared by the preparation step.

    Assigns an individual task to each intermediate file produced by the
    upstream step. Uses Moses reversible tokenizer to perform tokenization.
    """

    inp_file_path = Parameter()

    def __init__(self, *args, **kwargs):
        super(Tokenize, self).__init__(*args, **kwargs)
        self.act_out_dir_path = get_act_out_dir_path(
            GlobalConfig().out_dir_path,
            self.inp_file_path,
            FOLDER_NAME)

    def complete(self):
        """A naive check for completion."""
        if not os.path.isdir(self.act_out_dir_path):
            return False
        if not len(os.listdir(self.act_out_dir_path)):
            return False
        return True

    def output(self):
        return LocalTarget(self.act_out_dir_path)

    def requires(self):
        return [PrepareFile(inp_file_path=self.inp_file_path)]

    def run(self):
        # TODO: it seems that if I assign a file tokenizer to each file in a loop
        # TODO: as a dependency, it makes the whole system super slow if the
        # TODO: number of groups (=files) is very large

        init_logger(LOGGER_NAME)
        inp_folder = self.input()[0]
        failed_file_count = 0
        logger.info("Tokenizing `%s`." % inp_folder.path)
        for file_path in iter_file_paths(inp_folder.path):
            file_name = get_file_name(file_path)
            out_file_path = comb_paths(self.act_out_dir_path,
                                       "%s.csv" % file_name)
            try:
                # TODO: `_csv.Error: line contains NULL byte` was encountered in
                # TODO: a small number of files; the cause needs to be
                # TODO: investigated
                TokenizeFile(inp_file_path=file_path,
                             out_file_path=out_file_path).run()
            except CsvError:
                failed_file_count += 1
        logger.info('Failed to tokenize `%d` files in `%s`.' %
                    (failed_file_count, inp_folder.path))


class TokenizeFile(Task):
    """Tokenizes a single csv file."""

    inp_file_path = Parameter()
    out_file_path = Parameter()

    def complete(self):
        """Checks if the output file exists and if it's non-empty."""
        if not os.path.exists(self.out_file_path):
            return False
        return os.stat(self.out_file_path).st_size > 0

    def run(self):
        safe_mkfdir(self.out_file_path)
        data_units = []
        for data_unit in read_csv_file(self.inp_file_path, sep='\t'):
            tok_str = " ".join(tokenizer(data_unit[OutputFields.REV_TEXT]))
            data_unit[OutputFields.REV_TEXT] = tok_str
            data_units.append(data_unit)
        write_group_to_csv(self.out_file_path, data_units, sep='\t')
