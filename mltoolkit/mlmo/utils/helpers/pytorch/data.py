# contain data related modelling functions, mostly in PyTorch
import torch as T
from torch.autograd import Variable
import numpy as np


def create_mask_from_lens(lens):
    mask = T.zeros((lens.size(0), lens.max()), dtype=T.float32,
                   device=lens.device)
    for indx, l in enumerate(lens):
        mask[indx, :l] = 1.
    return mask


def adjust_tensor_to_beam_size(tens, beam_size):
    """Replicates tensor values for each beam over the first dim."""
    bs = tens.size(0)
    if len(tens.shape) == 3:
        s = tens.size(1)
        tens = tens.unsqueeze(1).repeat((1, beam_size, 1, 1))
        tens = tens.view(bs * beam_size, s, -1)
    elif len(tens.shape) == 2:
        s = tens.size(1)
        tens = tens.unsqueeze(1).repeat((1, beam_size, 1))
        tens = tens.view(bs * beam_size, s)
    elif len(tens.shape) == 1:
        tens = tens.unsqueeze(1).repeat((1, beam_size))
        tens = tens.view(bs * beam_size)
    else:
        raise ValueError("Wrong dim of att.")
    return tens


def length_to_mask(length, max_len=None, dtype=None):
    """length: B.
    return B x max_len.
    If max_len is None, then max of length will be used.
    """
    assert len(length.shape) == 1, 'Length shape should be 1 dimensional.'
    max_len = max_len or length.max().item()
    mask = T.arange(max_len, device=length.device,
                    dtype=length.dtype).expand(len(length),
                                               max_len) < length.unsqueeze(1)
    if dtype is not None:
        mask = T.as_tensor(mask, dtype=dtype, device=length.device)
    return mask


def reverse_padded_sequence(inputs, lengths, batch_first=False):
    """Reverses sequences according to their lengths.
    Inputs should have size ``T x B x *`` if ``batch_first`` is False, or
    ``B x T x *`` if True. T is the length of the longest sequence (or larger),
    B is the batch size, and * is any number of dimensions (including 0).
    Arguments:
        inputs (Variable): padded batch of variable length sequences.
        lengths (list[int]): list of sequence lengths
        batch_first (bool, optional): if True, inputs should be B x T x *.
    Returns:
        A Variable with the same size as inputs, but with each sequence
        reversed according to its length.
    """
    if batch_first:
        inputs = inputs.transpose(0, 1)
    max_length, batch_size = inputs.size(0), inputs.size(1)
    if len(lengths) != batch_size:
        raise ValueError('inputs is incompatible with lengths.')
    ind = [list(reversed(range(0, length))) + list(range(length, max_length))
           for length in lengths]
    ind = Variable(T.LongTensor(ind).transpose(0, 1))
    for dim in range(2, inputs.dim()):
        ind = ind.unsqueeze(dim)
    ind = ind.expand_as(inputs)
    if inputs.is_cuda:
        ind = ind.cuda(inputs.get_device())
    reversed_inputs = T.gather(inputs, 0, ind)
    if batch_first:
        reversed_inputs = reversed_inputs.transpose(0, 1)
    return reversed_inputs


def convert_tensors_to_numpy(chunk):
    """Converts each field value that is a Pytorch Tensor to numpy."""
    for fn, val in chunk.items():
        if isinstance(val, T.Tensor):
            val = chunk[fn].numpy()
        chunk[fn] = val
    return chunk
