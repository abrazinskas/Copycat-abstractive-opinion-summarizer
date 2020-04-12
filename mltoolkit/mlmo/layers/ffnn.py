from torch.nn import Module, Sequential, Linear, ReLU
from mltoolkit.mlutils.helpers.general import listify


class Ffnn(Module):
    """A feed-forward neural network that outputs scores \in [-inf, +inf]."""

    def __init__(self, input_dim, output_dim, hidden_dim=None,
                 non_linearity=None):
        """
        The network inputs a continues vector as input that at least contains
        the average embedding of aspects.

        :param hidden_dim: self-explanatory.
        :type hidden_dim: int or list of ints.
        :param output_dim: self-explanatory.
        :param non_linearity: an object for non-linear transformations.
        """
        super(Ffnn, self).__init__()
        hidden_dims = listify(hidden_dim) if hidden_dim is not None else None
        self._seq_model = Sequential()

        prev_hd = input_dim
        i = 0
        if hidden_dims is not None:
            for hd in hidden_dims:
                self._seq_model.add_module(str(i), Linear(prev_hd, hd))
                i += 1
                if non_linearity is not None:
                    self._seq_model.add_module(str(i), non_linearity)
                i += 1
                prev_hd = hd
        self._seq_model.add_module(str(i), Linear(prev_hd, output_dim))

    def forward(self, inp):
        """
        :param inp: [batch_size, input_dim]
        :return: [batch_size, output_dim]
        """
        output = self._seq_model(inp)
        return output
