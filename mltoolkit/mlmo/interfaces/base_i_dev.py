from .base_interface import BaseInterface
from .base_i_model import BaseIModel
from mltoolkit.mldp import Pipeline
from logging import getLogger
from mltoolkit.mlutils.helpers.formatting.general import format_big_box, \
    metrics_to_str
from collections import OrderedDict
import os
import warnings
from time import time

logger_name = os.path.basename(__file__)
logger = getLogger(logger_name)


class BaseIDev(BaseInterface):
    """
    Base class for modelling development interfaces. Interfaces of this type
    contain logic specific for development purposes, such as how to eval
    the model.
    """

    def __init__(self, imodel, train_data_pipeline, val_data_pipeline=None,
                 name_prefix=None):
        if not isinstance(imodel, BaseIModel):
            raise TypeError("Please provide a valid interface of a model.")
        if not isinstance(train_data_pipeline, Pipeline):
            raise ValueError("Please provide a valid training data pipeline.")
        if not isinstance(val_data_pipeline, Pipeline):
            raise ValueError("Please provide a valid validation data pipeline.")
        super(BaseIDev, self).__init__(name_prefix=name_prefix)
        self.imodel = imodel
        self.train_data_pipeline = train_data_pipeline
        self.val_data_pipeline = val_data_pipeline if val_data_pipeline else \
            train_data_pipeline

    def save_setup_str(self, dir_path, exper_descr=None):
        """
        Logs/saves the setup of the dev. experiment, namely 3 main components:
        1. Experiment's description -> experiment.txt (if provided exper_descr)
        2. dev. data pipeline's blueprint -> dp.txt
        3. model's blueprint/summary -> model.txt
        """
        logger.info("Experiment's output will be saved to: '%s'." % dir_path)
        # 1. experiment
        if exper_descr:
            form_exp = format_big_box(exper_descr)
            logger.info(form_exp)

        # 2. data pipeline
        dp_fp = os.path.join(dir_path, 'dp.txt')
        try:
            with open(dp_fp, 'w') as f:
                f.write(str(self.train_data_pipeline))
        except Exception:
            os.remove(dp_fp)
            warnings.warn(
                "Could not get the str of the train_data_pipeline's setup.")

        # 3. model and its interface
        m_fp = os.path.join(dir_path, 'model.txt')
        try:
            with open(m_fp, 'w') as f:
                f.write(str(self.imodel))
        except Exception:
            os.remove(m_fp)
            warnings.warn("Could not get the str of the model's setup.")

    def standard_workflow(self, train_data_source, val_data_source=None,
                          test_data_source=None, epochs=1,
                          logging_period=10, eval_train=False,
                          after_epoch_func=None):
        """
        Runs a workflow of steps such as training and evaluation. It executes
        a very general workflow, where eval flags can be assigned in order to
        perform evaluation on train, val, and test data-sources.
        
        :param train_data_source: self-explanatory.
        :param val_data_source: self-explanatory.
        :param test_data_source: self-explanatory.
        :param epochs: self-explanatory.
        :param logging_period: how often to log the loss of the model
        :param eval_train: whether to eval performance on the training
                           data-source.
        :param after_epoch_func: a function that takes as input 'epoch', and is
                                 executed after completion of each epoch, except
                                 the last one. E.g. model saving.
        """
        if val_data_source:
            metrs = self.eval_on_intern_metrs(data_source=val_data_source)
            logger.info(metrics_to_str(metrs, "Validation"))

        epoch = 0
        for epoch in range(1, epochs + 1):
            logger.info('Epoch %d/%d' % (epoch, epochs))
            self.train(data_source=train_data_source, epoch=epoch,
                       logging_steps=logging_period)

            if eval_train:
                metrs = self.eval_on_intern_metrs(data_source=train_data_source,
                                                  epoch=epoch)
                if metrs:
                    logger.info(metrics_to_str(metrs, "Training"))

            if val_data_source:
                metrs = self.eval_on_intern_metrs(data_source=val_data_source)
                logger.info(metrics_to_str(metrs, "Validation"))

            if epoch != epochs and after_epoch_func:
                after_epoch_func(epoch)

        if test_data_source:
            metrs = self.eval_on_intern_metrs(data_source=test_data_source,
                                              epoch=epoch)
            logger.info(metrics_to_str(metrs, "Testing"))

    def train(self, data_source, logging_steps=10, **kwargs):
        """
        Performs a single epoch training on the passed data_source.

        :param data_source: self-explanatory.
        :param logging_steps: self-explanatory.
        """
        logger.info("Training data source: %s" % data_source)
        start = time()
        for i, batch in enumerate(self.train_data_pipeline.iter(**data_source),
                                  1):
            metrics = self.imodel.train(batch=batch, **kwargs)
            if i % logging_steps == 0:
                mess = metrics_to_str(metrics, prefix="Chunk # %d" % i)
                logger.info(mess)
        logger.info("Epoch training time elapsed: %.2f (s)" % (time() - start))

    def eval_on_intern_metrs(self, data_source, **kwargs):
        """
        Runs the model for each batch and collects/accumulates its internal
        metrics (e.g.  loss, kld), which are assumed to be averaged over the
        number of data-units. Then aggregates by the total number of data-units
        division.
        """
        logger.info("Evaluation data source: %s" % data_source)
        total_metrs = OrderedDict()
        total_dus = 0
        start = time()

        for i, batch in enumerate(self.val_data_pipeline.iter(**data_source),
                                  1):
            metrs = self.imodel.eval(batch=batch, **kwargs)
            for k, v in metrs.items():
                if k not in total_metrs:
                    total_metrs[k] = 0.
                total_metrs[k] += v * len(batch)  # rescaling back
            total_dus += len(batch)

        logger.info("Evaluation time elapsed: %.2f (s)" % (time() - start))

        # compute the actual average over data-units
        f_res = OrderedDict()
        for k, v in total_metrs.items():
            f_res[k] = v / float(total_dus)

        return f_res
