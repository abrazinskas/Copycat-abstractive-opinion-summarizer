from mltoolkit.mldp.steps.transformers import BaseTransformer
import numpy as np


class InvalidTransformer(BaseTransformer):

    def _transform(self, data_chunk):
        return np.random.random((10, 5))
