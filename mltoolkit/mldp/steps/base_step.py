from mltoolkit.mlutils.tools import SignedObject


class BaseStep(SignedObject):
    """Base class of all steps used in the Pipeline class."""

    def __init__(self, name_prefix=None):
        """
        :param name_prefix: a str that will prefix the title of the object if
                            the signature is generated.
        """
        super(BaseStep, self).__init__(name_prefix=name_prefix)
