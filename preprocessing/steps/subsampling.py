from luigi import Task, Parameter, IntParameter, LocalTarget
from preprocessing.global_config import GlobalConfig
from preprocessing.fields import OutputFields
from mltoolkit.mlutils.helpers.paths_and_files import iter_file_paths, \
    get_file_name, \
    comb_paths
from preprocessing.helpers.data_utils import read_csv_file, \
    get_act_out_dir_path, write_groups_to_csv
from mltoolkit.mlutils.helpers.formatting.general import format_stats
from mltoolkit.mlutils.helpers.logging import init_logger
from preprocessing.steps import Tokenize
import os
from logging import getLogger
import numpy as np
from collections import OrderedDict

FOLDER_NAME = "3.subsam"
LOGGER_NAME = os.path.basename(__file__)
logger = getLogger(LOGGER_NAME)


class Subsample(Task):
    """Subsampling of one category of reviews.

    Filters out too long/short reviews; unpopular groups. Then, filters out
    groups that are above a percentile review count and optionally makes sures
    that the passed threshold number of reviews is not exceeded.
    """

    inp_file_path = Parameter()

    min_revs = IntParameter()
    min_rev_len = IntParameter(default=10,
                               description='Minimum number of reviews a group should have.')
    max_rev_len = IntParameter(default=70,
                               description='Maximum number of tokens a review should have.')
    percentile = IntParameter(default=90,
                              description='Threshold percentile of reviews, if a groups with higher number of reviews are discarded.')
    max_total_revs = IntParameter(default=None,
                                  description='Limits the total number of reviews that the step outputs per file.')

    def __init__(self, *args, **kwargs):
        super(Subsample, self).__init__(*args, **kwargs)
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
        return [Tokenize(inp_file_path=self.inp_file_path)]

    def run(self):
        log_file_path = comb_paths(GlobalConfig().out_dir_path,
                                   "logs", FOLDER_NAME,
                                   "%s.txt" % get_file_name(self.inp_file_path))
        init_logger(LOGGER_NAME, output_path=log_file_path)

        init_unit_count = 0
        init_group_count = 0

        group_id_to_units = {}
        group_unit_counts = []

        # 1. reading data and filtering out short/long reviews and
        # unpopular groups
        inp_dir = self.input()[0]
        logger.info("Subsampling `%s`." % inp_dir.path)
        for inp_file_path in iter_file_paths(inp_dir.path):
            group_id = get_file_name(inp_file_path)
            group_units = []
            init_group_count += 1
            for data_unit in read_csv_file(inp_file_path, sep='\t'):
                init_unit_count += 1
                rev_text = data_unit[OutputFields.REV_TEXT].split()

                # removing too short and long reviews
                if len(rev_text) < self.min_rev_len or \
                        len(rev_text) > self.max_rev_len:
                    continue

                group_units.append(data_unit)

            # removing unpopular groups
            if len(group_units) < self.min_revs:
                continue

            group_id_to_units[group_id] = group_units
            group_unit_counts.append(len(group_units))

        if not len(group_id_to_units):
            raise ValueError("No groups to proceed.")

        # 2. filtering by percentile
        perc = np.percentile(group_unit_counts, self.percentile)

        # removing above a kth percentile groups
        subs_group_id_to_units = {}
        subs_units_count = 0
        subs_units_max_count = 0

        for group_id, group_units in group_id_to_units.items():
            if len(group_units) < perc:

                # making sure that the subsampled number of reviews does not
                # exceed a threshold
                if self.max_total_revs is not None \
                        and (subs_units_count + len(
                    group_units)) > self.max_total_revs:
                    break

                subs_units_count += len(group_units)
                subs_units_max_count = max(subs_units_max_count,
                                           len(group_units))

                subs_group_id_to_units[group_id] = group_units

        if subs_units_count == 0:
            raise ValueError("All units were subsampled out. "
                             "Please adjust the parameters.")

        # 3. dumping to files
        write_groups_to_csv(self.act_out_dir_path, subs_group_id_to_units,
                            sep='\t')

        # 4. logging statistics
        stats = OrderedDict()
        stats['General'] = OrderedDict()
        stats['General']['inp dir'] = inp_dir.path

        stats['Initial'] = OrderedDict()
        stats['Initial']['group count'] = init_group_count
        stats['Initial']['unit count'] = init_unit_count

        stats['After Filtering'] = OrderedDict()
        stats['After Filtering']['group count'] = len(group_id_to_units)
        stats['After Filtering']['unit count'] = np.sum(group_unit_counts)
        stats['After Filtering']['percentile count'] = perc

        stats['After Subsampling'] = OrderedDict()
        stats['After Subsampling']['group count'] = len(subs_group_id_to_units)
        stats['After Subsampling']['unit count'] = subs_units_count
        stats['After Subsampling']['max units per group'] = subs_units_max_count

        stats_str = format_stats(stats)

        logger.info(stats_str)
