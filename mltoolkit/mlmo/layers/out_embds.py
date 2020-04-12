from torch.nn import Module
import torch as T


class OutEmbds(Module):
    """
    This module re-uses a fixed number of embeddings from a pre-existing module.
    Can be used to share embeddings across the system.
    """

    def __init__(self, inp_embds, vocab_size):
        super(OutEmbds, self).__init__()
        self.embds = inp_embds
        self.vocab_size = vocab_size

    def forward(self, x):
        out = T.matmul(x, self.embds.weight[:self.vocab_size].t())
        return out
