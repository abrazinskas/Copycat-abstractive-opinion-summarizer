from mltoolkit.mldp.steps.base_step import BaseStep


class BasePreProcessor(BaseStep):
    """
    Subclasses of this step allow to execute custom logic before data processing
    starts. For example, downloading data to the local storage, or shuffling.

    A Preprocessor does not have a pre-specified output format. But the output
    will be fed to the reader, so input-output must be coherent.
    """

    def __init__(self, **kwargs):
        super(BasePreProcessor, self).__init__(**kwargs)

    def __call__(self, **kwargs):
        """
        This is the main method that suppose to execute the actual pre-processing
        logic, and pass the output along the pipeline to the reader.
        :return: a reader coherent output, e.g. file paths.
        """
        raise NotImplementedError
