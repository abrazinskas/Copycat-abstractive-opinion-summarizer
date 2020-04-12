from mltoolkit.mldp import Pipeline
from mltoolkit.mldp.steps.formatters import PyTorchFormatter


class PyTorchPipeline(Pipeline):
    """
    Addresses the issue associated with processes that put PyTorch tensors to a
    Queue object must be alive when the main processes gets/requests them from
    the Queue. Formats batches on the main processes.

    https://discuss.pytorch.org/t/multiprocessing-using-torch-multiprocessing/11029
    https://discuss.pytorch.org/t/using-torch-tensor-over-multiprocessing-queue-process-fails/2847/2
    """

    def __init__(self, **kwargs):
        super(PyTorchPipeline, self).__init__(**kwargs)
        self.torch_formatter = PyTorchFormatter()

    def iter(self, early_term=None, **kwargs):
        """For more info see the original method."""
        itr = super(PyTorchPipeline, self).iter(early_term=early_term, **kwargs)
        for dc in itr:
            yield self.torch_formatter(dc)
