from torch.nn.parallel import DataParallel


def parallelize_submodules(parent_module):
    """
    Wraps a parent's sub-modules with a DataParallel object that permits multi
    GPU usage.
    """
    for mod_name, module in parent_module.named_children():
        setattr(parent_module, mod_name, DataParallel(module))
