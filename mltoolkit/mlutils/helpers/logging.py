import logging
import os
from mltoolkit.mlutils.helpers.paths_and_files import safe_mkfdir, safe_mkdir, \
    is_file_path
from time import strftime
from logging import NOTSET, FATAL, DEBUG, INFO


def function_logging_decorator(logger, log_args=True, log_kwargs=True,
                               class_name="", logger_func='debug'):
    """Creates a decorator specific to a passed logger. Used to wrap funcs."""

    def decorator(func):
        def wrapper(*args, **kwargs):
            mess = "Executing %s" % func.__name__
            if class_name != "":
                mess += " of %s" % class_name
            if log_args:
                mess += ", args: %s" % args
            if log_kwargs:
                mess += ", kwargs: %s" % kwargs
            getattr(logger, logger_func)(mess)
            result = func(*args, **kwargs)
            return result

        return wrapper

    return decorator


def init_logger(logger_name=__name__, output_path=None, level=logging.INFO):
    """
    Initializes a logger for console and file writing.

    :param logger_name: self-explanatory.
    :param output_path: directory or file path where the logs should be saved.
                        By default it will not store file logs.
    :param level: self-explanatory.
    """
    logger = logging.getLogger(logger_name)
    logger.setLevel(level)

    formatter = logging.Formatter("%(asctime)s [%(levelname)s]: %(message)s")

    # adding console output
    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(formatter)
    logger.addHandler(stream_handler)

    if output_path:
        if is_file_path(output_path):
            safe_mkfdir(output_path)
            if os.path.exists(output_path):
                os.remove(output_path)
        else:
            safe_mkdir(output_path)
            # using the default name of the logger
            default_file_name = "log_" + strftime("%b_%d_%H_%M_%S") + '.txt'
            output_path = os.path.join(output_path, default_file_name)
        file_handler = logging.FileHandler(output_path)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    return logger
