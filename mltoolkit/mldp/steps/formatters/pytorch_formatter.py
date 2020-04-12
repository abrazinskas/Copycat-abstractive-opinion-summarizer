from mltoolkit.mldp.steps.formatters import BaseFormatter
import numpy as np
import torch as T


class PyTorchFormatter(BaseFormatter):
    """Converts numpy arrays of integers and floats to PyTorch tensors."""

    def _format(self, data_chunk):
        allowed_types = [np.int32, np.int64, np.float32, np.float64]
        for k, v in data_chunk.items():
            if isinstance(v, np.ndarray) and v.dtype in allowed_types:
                v = self._convert_to_tensors(v)
            data_chunk[k] = v
        return data_chunk

    def _convert_to_tensors(self, v):
        """Fixes problems with torch incomp. dtypes. and coverts to tensors."""
        if not isinstance(v, np.ndarray):
            v = np.array(v)
        # recasting types to avoid later conflicts
        if v.dtype == 'int32':
            v = v.astype('int64')
        if v.dtype == 'float64':
            v = v.astype('float32')
        t = T.from_numpy(v)
        return t
