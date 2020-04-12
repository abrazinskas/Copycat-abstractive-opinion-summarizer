import torch as T
from torch.nn import Module
from mltoolkit.mlmo.layers import Ffnn


class MuSigmaFfnn(Module):
    """
    A single hidden layer feed-forward nn that outputs mu and sigma of a
    Gaussian distribution. The produced sigma is a vector with non-negative
    values.
    """

    def __init__(self, input_dim, output_dim, hidden_dim=None, **kwargs):
        """
        The network inputs a continues vector as input that at least contains
        the average embedding of aspects.

        :param input_dim: self-explanatory.
        :param hidden_dim: self-explanatory.
        :type hidden_dim: int or list of ints.
        :param output_dim: dimensionality of mus and sigmas.
        """
        super(MuSigmaFfnn, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.hidden_dim = hidden_dim

        self.ffnn = Ffnn(input_dim=input_dim, hidden_dim=hidden_dim,
                         output_dim=2 * output_dim, **kwargs)

    def forward(self, inp):
        """
        :param inp: [batch_size, input_dim]
        :return: mu [batch_size, output_dim]
                 sigma [batch_size, output_dim]
        """
        out = self.ffnn(inp)
        mu = out[:, :self.output_dim]
        log_sigma = out[:, self.output_dim:]
        return mu, T.exp(log_sigma)
