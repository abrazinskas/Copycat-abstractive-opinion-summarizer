from logging import getLogger
from mltoolkit.mlmo.utils.helpers.loading_and_saving import load_params_from_pkl
from mltoolkit.mlmo.interfaces.base_interface import BaseInterface
import os

logger_name = os.path.basename(__file__)
logger = getLogger(logger_name)


class BaseIModel(BaseInterface):
    """
    Interface over models that contain general wrapper methods, such as
    training and prediction over batches.
    """

    def __init__(self, model, **kwargs):
        super(BaseIModel, self).__init__(**kwargs)
        self._model = model

    @property
    def model(self):
        return self._model

    @model.setter
    def model(self, model):
        self._model = model

    def load_params(self, params_dump_file_path, excluded_params=None):
        """
        Loads parameters from a pickle _dump file.

        :param params_dump_file_path: self-explanatory
        :param excluded_params: a list of parameter names that should NOT be
                               initialized.
        """
        params_dict = load_params_from_pkl(params_dump_file_path)
        logger.info("Loaded parameters from: %s" % params_dump_file_path)
        for k, v in params_dict.items():
            if excluded_params is not None and k in excluded_params:
                continue
            self.model.initialize_param(k, v)
        logger.info("Initialized the following parameters: %s" %
                    (", ".join(params_dict.keys())))

    def __str__(self):
        """
        Returns a human readable str repr. of the model and its interface if it
        contains some valuable information."""
        raise NotImplementedError

    def train(self, **kwargs):
        """
        Performs training on a batch of data, returns a dictionary with metrics
        evaluated on the batch.
        """
        raise NotImplementedError

    def eval(self, **kwargs):
        """
        Computes internal model's metrics over a batch. Returns a dictionary of
        their average over the number of data-units.
        """
        raise NotImplementedError

    def predict(self, **kwargs):
        raise NotImplementedError
