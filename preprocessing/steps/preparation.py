from luigi import Task, LocalTarget, Parameter
from mltoolkit.mlutils.helpers.paths_and_files import get_file_paths, comb_paths
from preprocessing.global_config import GlobalConfig
from preprocessing.helpers.data_utils import read_amazon_data, \
    read_yelp_data, write_group_to_csv, get_act_out_dir_path
import os
from logging import getLogger
from mltoolkit.mlutils.helpers.logging import init_logger

FOLDER_NAME = "1.prep"
LOGGER_NAME = os.path.basename(__file__)
logger = getLogger(LOGGER_NAME)


class PrepareFile(Task):
    """Enriches and formats data in a file; splits groups to interm. CSV files.

    Notes:
        Each data-unit(review+attributes) will have the category attribute
        that will take as value the name of the input file.

    """

    inp_file_path = Parameter()

    def __init__(self, *args, **kwargs):
        super(PrepareFile, self).__init__(*args, **kwargs)
        self.act_out_dir_path = get_act_out_dir_path(
            GlobalConfig().out_dir_path,
            self.inp_file_path,
            FOLDER_NAME)

    def complete(self):
        """A naive test for whether to rerun the task."""
        if not os.path.isdir(self.act_out_dir_path):
            return False
        if not len(os.listdir(self.act_out_dir_path)):
            return False
        return True

    def output(self):
        return LocalTarget(self.act_out_dir_path)

    def run(self):
        init_logger(LOGGER_NAME)
        logger.info("Preparing `%s`." % self.inp_file_path)
        if GlobalConfig().dataset == 'amazon':
            iter = read_amazon_data(self.inp_file_path, replace_xml=True)
        else:
            iter = read_yelp_data(self.inp_file_path)
        for group_id, dus in iter:
            full_file_name = "%s.csv" % group_id
            out_file_path = comb_paths(self.act_out_dir_path, full_file_name)
            write_group_to_csv(out_file_path, dus, sep='\t')
