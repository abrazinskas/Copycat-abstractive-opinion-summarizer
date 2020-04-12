from luigi import Config, Parameter
from mltoolkit.mlutils.helpers.paths_and_files import comb_paths


class GlobalConfig(Config):
    out_dir_path = Parameter()
    dataset = Parameter(default='amazon')

    def __init__(self, *args, **kwargs):
        super(GlobalConfig, self).__init__(*args, **kwargs)
        self.out_dir_path = comb_paths(self.out_dir_path, self.dataset)
