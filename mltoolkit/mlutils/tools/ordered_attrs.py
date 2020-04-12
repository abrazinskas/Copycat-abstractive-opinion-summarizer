from collections import OrderedDict
from types import MethodType


# def setattr_decor(obj, fn):
#     def dec(key, value):
#         print key
#         print value
#
#         return fn(key, value)
#     return dec


class OrderedAttrs(object):
    """
    Allows to store object attributes in the order of assignment in a sep var.
    Used for automatic information scraping of objects.
    """

    def __init__(self):
        super(OrderedAttrs, self).__init__()
        self.__odict__ = OrderedDict()

    def __setattr__(self, key, value):
        if key != "__odict__":
            self.__odict__[key] = value
        return super(OrderedAttrs, self).__setattr__(key, value)

    def func(self, a):
        pass
