from mltoolkit.mlutils.helpers.paths_and_files import safe_mkfdir
from collections import OrderedDict
import os
from logging import getLogger
import json
import codecs
from argparse import ArgumentParser, ArgumentTypeError
from mltoolkit.mlutils.helpers.argparse import str2bool, str2list, \
    str2floatlist, str2intlist

OVERRIDABLE_ATTR_TYPES = (list, int, float, bool, str, dict)
CONSOLE_ATTR_TYPES = (list, int, float, bool, str)
FLOAT_FORMAT = "%.3f"
MAX_LINE_LEN = 70
logger_name = os.path.basename(__file__)
logger = getLogger(logger_name)


class BaseHP(object):
    """
    Children objects of this class can be used to store default hyper-parameters
    override then when necessary through loading, and then save them to a json
    file.

    Not all hyper-param fields can be saved and loaded, only of simple types,
    such as string, integer, boolean, float, list, etc.
    """

    def __init__(self, excl_print_attrs=None):
        """
        Args:
            excl_print_attrs (list): list of attribute names to be
                excluded from __str__.
        """
        super(BaseHP, self).__init__()
        if excl_print_attrs is None:
            excl_print_attrs = []
        else:
            if not isinstance(excl_print_attrs, list):
                raise TypeError("Please provide list for `excl_print_attrs`.")
        self._excl_from_print = []
        for attr_name in excl_print_attrs:
            self._excl_from_print.append(attr_name)

    def save(self, file_path, encoding='utf-8'):
        """Saves hyper-params object as a json file."""
        safe_mkfdir(file_path)
        params = self.to_dict(types_whitelist=OVERRIDABLE_ATTR_TYPES)
        f = codecs.open(file_path, encoding=encoding, mode='w')
        json.dump(params, f, indent=2)
        logger.debug("Extracted the following hparams: '%s'."
                     "" % " ".join(params.keys()))
        logger.info("Saved hyper-parameters to '%s'." % file_path)

    def load(self, file_path, encoding='utf-8'):
        """Loads hyper-params from a json file."""
        f = codecs.open(file_path, encoding=encoding)
        hparams = json.load(f, encoding=encoding)
        self.override(hparams)
        logger.info("Loaded hyper-parameters from '%s'." % file_path)

    def override(self, dct, error_on_mismatch=True):
        """Overrides field values based on the passed dictionary."""
        for attr_name, attr_val in dct.items():
            if not hasattr(self, attr_name):
                if error_on_mismatch:
                    msg = "Could not override '%s' because the field does " \
                          "not exist." % attr_name
                    logger.error(msg)
                    raise ValueError(msg)
                continue
            setattr(self, attr_name, attr_val)

    def add_parser_args(self, parser):
        """
        Traverses attributes of the object, adds them to the parser as
        arguments.
        """
        assert isinstance(parser, ArgumentParser)

        def get_list_conv_func(val):
            if len(val) == 0:
                tp = int
            else:
                tp = type(val[0])
                assert all([type(v) == tp for v in val])
            if tp == int:
                return str2intlist
            elif tp == float:
                return str2floatlist
            else:
                return str2list

        params = self.to_dict(types_whitelist=CONSOLE_ATTR_TYPES)
        for name, default_val in params.items():
            param_type = type(default_val)
            if param_type == bool:
                param_type = str2bool
            elif param_type == list:
                param_type = get_list_conv_func(default_val)
                default_val = ",".join([str(v) for v in default_val])
            param_type_name = param_type.__name__
            class_name = self.__class__.__name__
            # add extra handling of lists
            descr = "class=%s, type=%s, default=%s" % (class_name,
                                                       param_type_name,
                                                       default_val)
            parser.add_argument("--%s" % name, type=param_type,
                                default=default_val, help=descr)

    def to_dict(self, types_whitelist=None, names_blacklist=None):
        """
        Converts the object to a dictionary, excludes protected and additional
        ones (optionally).
        """
        res = OrderedDict()
        for attr_name, attr_val in vars(self).items():
            # excluding based on blacklisted names
            if names_blacklist is not None and attr_name in names_blacklist:
                continue
            # excluding based on protected names
            if attr_name[0] == "_":
                continue
            # include only specific types
            if types_whitelist is not None and \
                    not isinstance(attr_val, types_whitelist):
                continue
            res[attr_name] = attr_val
        return res

    def __str__(self):
        """
        Creates a string from the object for logging and printing.
        Lines are splat by the new line based on the maximum length
        chars (constant).
        """
        lines = []
        curr_line_len = 0
        curr_attrs = []
        params = self.to_dict(types_whitelist=OVERRIDABLE_ATTR_TYPES,
                              names_blacklist=self._excl_from_print)
        for name, val in params.items():
            if curr_line_len >= MAX_LINE_LEN:
                lines.append(", ".join(curr_attrs))
                curr_line_len = 0
                curr_attrs = []
            if isinstance(val, float):
                template = "%s: " + FLOAT_FORMAT
            else:
                template = "%s: %s"
            name_val_str = template % (name, val)
            curr_line_len += len(name_val_str)
            curr_attrs.append(name_val_str)
        if curr_line_len > 0:
            lines.append(", ".join(curr_attrs))
        lines_str = "\n".join(lines)
        return lines_str
