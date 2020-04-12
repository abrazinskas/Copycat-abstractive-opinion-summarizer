from types import FunctionType, MethodType, LambdaType, BuiltinFunctionType
from collections import OrderedDict
from mltoolkit.mlutils.helpers.general import listify, is_custom_object


class SignatureScraper(object):
    """
    Responsible for production of object signatures that can be later used for
    production of automatic documentation of objects.
    """

    def __init__(self, extra_allowed_types=None, excl_types=None,
                 excl_attr_names=None, scrape_obj_vals=False):
        """
        :param extra_allowed_types: list of attr value types which should be
                                    added to the default ones.
        :param excl_types: list of attr value types, which should not be
                           ignored. Namely, if an attribute value if of the
                           excluded type - the attribute is ignored.
        :param excl_attr_names: list of attribute names that should not be
                                scrapped.
        :param scrape_obj_vals: whether to represent attr values which are
                                custom objects.
        :return: a dictionary of attribute key-value pairs.
        """
        self.allowed_types = (int, str, float, list, dict, bool,
                              OrderedDict, FunctionType, MethodType, LambdaType,
                              BuiltinFunctionType)

        self.adjust_allowed_types(extra_allowed_types, excl_types)
        self.excl_attr_names = excl_attr_names
        self.scrape_obj_vals = scrape_obj_vals

    def adjust_allowed_types(self, ext_allowed_types=None, excl_types=None):
        """
        Adjusts the already allowed types by including extra ones and excluding
        provided ones. Sets the allowed_types attribute to a new tuple.
        """
        allowed_types = set()
        ext_allowed_types = ext_allowed_types if ext_allowed_types else set()
        excl_types = excl_types if excl_types else set()

        for t in self.allowed_types:
            allowed_types.add(t)
        for t in ext_allowed_types:
            allowed_types.add(t)
        for t in excl_types:
            if t in allowed_types:
                allowed_types.remove(t)
        return tuple(allowed_types)

    def scrape(self, obj):
        """
        Extracts attributes from objects that potentially represent the object
        itself. This function is used to create automatic documentation of steps
        in the pipeline.

        Lambda functions are not represented beyond bare 'lambda' strs due to
        complexity of obtaining a better representation.

        :param obj: an object from which to scrape signature.
        """
        collector = OrderedDict()
        d = obj.__odict__ if hasattr(obj, "__odict__") else obj.__dict__
        for name, value in d.items():
            if name in self.excl_attr_names or self._is_prot_attr(name):
                continue
            try:
                collector[name] = self._repr_val(value)
            except ValueError:
                pass
        return collector

    def _repr_val(self, val):
        """
        Attempts to represent the value, raises an error if fails.
        Certain value types have their own rules regarding representation.
        """
        error = ValueError("Can't represent the value.")

        if not self._valid_val_type(val):
            # 0. Custom objects
            if self.scrape_obj_vals and is_custom_object(val):
                if hasattr(val, 'get_signature'):
                    _, attrs = val.get_signature()
                else:
                    attrs = self.scrape(val)
                return "%s(%s)" % (val.__class__.__name__, repr_dict(attrs))
            else:
                raise error

        # 2. Methods and standard functions
        if self._is_of_allowed_types(val,
                                     target_types=[FunctionType, MethodType,
                                                   BuiltinFunctionType,
                                                   LambdaType]):
            return repr_func(val)

        # 3. Lists
        if self._is_of_allowed_types(val, list):
            res = []
            for el in val:
                try:
                    res.append(self._repr_val(el))
                except ValueError:
                    pass
            if len(res):
                return "[" + ", ".join(res) + "]"
            else:
                raise error

        # 4. Dicts and OrderedDicts
        is_allowed_dict = self._is_of_allowed_types(val, dict)
        is_allowed_o_dict = self._is_of_allowed_types(val, OrderedDict)
        if is_allowed_dict or is_allowed_o_dict:
            if not len(val):
                return repr_dict(val)
            res = dict() if is_allowed_dict else OrderedDict()
            for k, v in val.items():
                try:
                    res[k] = self._repr_val(v)
                except ValueError:
                    pass
            if len(res):
                return repr_dict(res)
            # if it fails to represent dict values - will return keys only
            return "(keys)%s" % ("{" + ", ".join(val.keys()) + "}")

        return repr(val)

    def _is_of_allowed_types(self, val, target_types):
        """Checks if val belongs to specific target types (if allowed)."""
        target_types = listify(target_types)
        return any([t in self.allowed_types and isinstance(val, t) for
                    t in target_types])

    def _valid_val_type(self, val):
        return isinstance(val, self.allowed_types) or val is None

    @staticmethod
    def _is_prot_attr(attr_name):
        return attr_name[0] == '_'


def repr_dict(d):
    tmp_k_v_str = [("%s: %s" % (k, v)) for k, v in d.items()]
    return "{" + ", ".join(tmp_k_v_str) + "}"


def repr_func(func):
    """Attempts to return a representative document of a function/method."""
    try:
        if hasattr(func, "im_self"):
            im_self = func.im_self
            full_class_name = str(im_self.__class__)
            func_name = func.__name__
            return ".".join([full_class_name, func_name])
        else:
            return func.__name__
    except Exception:
        return str(func)
