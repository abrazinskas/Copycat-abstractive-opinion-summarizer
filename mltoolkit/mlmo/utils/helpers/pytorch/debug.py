import torch as T
from torch.nn import Parameter
import gc
import resource


def get_alive_tensors():
    """Returns tuples (type, size)."""
    params = []
    tensors = []
    for obj in gc.get_objects():
        if T.is_tensor(obj) or (hasattr(obj, 'data') and
                                T.is_tensor(obj.data)):
            tpl = (type(obj), obj.size(), obj.device)
            if isinstance(obj, Parameter):
                params.append(tpl)
            else:
                tensors.append(tpl)
    return params, tensors


def get_memory_consumption():
    """Return the number of MB for currently memory consumption."""
    gc.collect()
    max_mem_used = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return max_mem_used / float(1024)
