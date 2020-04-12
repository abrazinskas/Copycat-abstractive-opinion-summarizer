from mltoolkit.mlutils.tools import SignedObject


class BaseInterface(SignedObject):
    """Base class for all interfaces."""

    def __init__(self, name_prefix=None):
        super(BaseInterface, self).__init__(name_prefix=name_prefix)
