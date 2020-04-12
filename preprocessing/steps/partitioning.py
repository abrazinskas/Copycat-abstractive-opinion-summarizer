from luigi import Task, Parameter, FloatParameter, IntParameter
from preprocessing.steps import Subsample
from preprocessing.fields import AmazonFields, YelpFields, OutputFields
from preprocessing.global_config import GlobalConfig
from preprocessing.helpers.data_utils import partition
from mltoolkit.mlutils.helpers.paths_and_files import get_file_paths, \
    get_file_name
from preprocessing.helpers.data_utils import read_csv_file, \
    write_group_to_csv, comb_paths
from mltoolkit.mlutils.helpers.formatting.general import format_stats
from mltoolkit.mlutils.helpers.logging import init_logger
import os
from logging import getLogger
from collections import OrderedDict
import numpy as np

FOLDER_NAME = '4.part'
LOGGER_NAME = os.path.basename(__file__)
logger = getLogger(LOGGER_NAME)


class Partition(Task):
    """Splits data to training and test partitions, excludes some groups."""

    train_part = FloatParameter()
    val_part = FloatParameter(default=0.0)
    test_part = FloatParameter(default=0.0)

    inp_dir_path = Parameter()

    def __init__(self, *args, **kwargs):
        super(Partition, self).__init__(*args, **kwargs)
        self.act_out_dir_path = os.path.join(GlobalConfig().out_dir_path,
                                             FOLDER_NAME)

    def complete(self):
        return False

    def requires(self):
        tasks = []
        fps = get_file_paths(self.inp_dir_path)
        for fp in fps:
            tasks.append(Subsample(fp))
        return tasks

    def run(self):

        log_file_path = comb_paths(GlobalConfig().out_dir_path, "logs",
                                   "partitioning.txt")
        init_logger(LOGGER_NAME, output_path=log_file_path)

        excluded_group_count = 0
        list_group_units = []

        # tracking duplicate groups as one group can be in multiple categories
        group_ids = set()
        dup_group_count = 0

        curr_unit_count = 0
        # reading data and excluding some groups
        inp_dirs = []
        for inp_dir in self.input():
            inp_dirs.append(inp_dir.path)
            for inp_group_file_path in get_file_paths(inp_dir.path):
                group_id = get_file_name(inp_group_file_path)
                if group_id in group_ids:
                    dup_group_count += 1
                    continue
                group_ids.add(group_id)
                if self._is_excluded(group_id):
                    excluded_group_count += 1
                    continue
                units = [u for u in
                         read_csv_file(inp_group_file_path, sep='\t')]
                list_group_units.append(units)

        # partitioning
        logger.info("Partitioning `%s`." % " ".join([idp for idp in inp_dirs]))
        tr_part, \
        val_part, \
        test_part = partition(list_group_units, train_part=self.train_part,
                              val_part=self.val_part, test_part=self.test_part)

        # dumping to the storage
        for title, part in zip(['train', 'val', 'test'],
                               [tr_part, val_part, test_part]):
            if len(part):
                for group_units in part:
                    group_id = group_units[0][OutputFields.GROUP_ID]
                    group_file_path = comb_paths(self.act_out_dir_path, title,
                                                 '%s.csv' % group_id)
                    write_group_to_csv(group_file_path, group_units, sep='\t')

        # logging stats
        train_rev_count = np.sum([len(gr) for gr in tr_part])
        val_rev_count = np.sum([len(gr) for gr in val_part])
        test_rev_count = np.sum([len(gr) for gr in test_part])

        stats = OrderedDict()
        stats['General'] = OrderedDict()
        stats['General']['excluded_group_count'] = excluded_group_count
        stats['General']['duplicate_group_count'] = dup_group_count
        stats['General']['train_groups'] = len(tr_part)
        stats['General']['train_rev_count'] = train_rev_count
        stats['General']['val_groups'] = len(val_part)
        stats['General']['val_rev_count'] = val_rev_count
        stats['General']['test_groups'] = len(test_part)
        stats['General']['test_rev_count'] = test_rev_count

        logger.info(format_stats(stats))

    def _is_excluded(self, group_id):
        if GlobalConfig().dataset == 'amazon':
            return group_id in AmazonFields.EXCLUDED_GROUP_IDS
        if GlobalConfig().dataset == 'yelp':
            return group_id in YelpFields.EXCLUDED_GROUP_IDS
