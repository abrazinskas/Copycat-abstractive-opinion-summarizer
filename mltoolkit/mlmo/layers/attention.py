import torch as T
from torch.nn import Parameter
from torch.nn import Module, Linear, Tanh
from logging import getLogger
import os
from mltoolkit.mlmo.utils.helpers.pytorch.computation import masked_softmax

logger_name = os.path.basename(__file__)
logger = getLogger(logger_name)


class Attention(Module):
    """
    Attention mechanism is used to compute context vectors for queries by
    summing scaled values. Scores are computed by a single hidden layer neural
    network.

    """

    def __init__(self, query_dim, value_dim, hidden_dim, non_linearity=Tanh()):
        """
        :param query_dim:
        :param value_dim:
        :param hidden_dim:
        :param non_linearity: non-linearity of the score network that is used
                              after the hidden layer is computed.
        """
        super(Attention, self).__init__()
        self.query_dim = query_dim
        self.value_dim = value_dim
        self.hidden_dim = hidden_dim
        self.keys_lin_proj = Linear(value_dim, hidden_dim, bias=False)
        self.score_network = ScoreNetwork(query_dim=query_dim,
                                          hidden_dim=hidden_dim,
                                          non_linearity=non_linearity)

    def forward(self, query, value, key=None, mask=None, local_weights=None,
                tempr=None):
        """
        Optionally, performs local normalization of by passing `local_weights`
        parameter, such that weights of a group sum to 1.

        :param query: [batch_size, query_dim]
        :param key: [batch_size, seq_len, hidden_dim] if passed, otherwise
                    it will be created by linear projection of values.
        :param value: [batch_size, seq_len, value_dim]
        :param mask: [batch_size, seq_len]
        :param local_weights: [batch_size, local_len]
        :param tempr: scalar temperature by which scores are divided. Used for
                      making distributions more or less peaked.
        :return: contxt_vec: [batch_size, value_dim]
                 att_weights: [batch_size, seq_len]
        """
        assert len(value.shape) == 3
        assert len(query.shape) == 2
        if self.value_dim is None and key is None:
            raise ValueError("Please provide 'value_dim' to the constructor in "
                             "order to internally create keys.")
        batch_size = value.size(0)
        seq_len = value.size(1)

        if key is None:
            key = self.create_keys(value)

        scores = self.score_network(query=query, key=key)

        if tempr is not None:
            scores /= tempr

        if local_weights is not None:
            # performing local rescaling of weights
            local_len = local_weights.size(1)
            scores = scores.view(batch_size, local_len, seq_len / local_len)
            att_weights = masked_softmax(scores, mask=mask, dim=2)
            att_weights = att_weights * local_weights.unsqueeze(-1)
            att_weights = att_weights.view(batch_size, -1, 1)
        else:
            att_weights = masked_softmax(scores, mask=mask, dim=1)

        contxt_vec = (value * att_weights).sum(dim=1)
        att_weights = att_weights.squeeze(-1)

        return contxt_vec, att_weights

    def create_keys(self, vals):
        return self.keys_lin_proj(vals)


class ScoreNetwork(Module):
    """
    An optimized single hidden layer neural network for attention scores.
    The optimization idea behind this network is that projection of keys can
    performed only once without concatenation with query.

    It's allows to avoid unnecessary extra computations when attending every
    time-step over the same key-value pairs.
    """

    def __init__(self, query_dim, hidden_dim, non_linearity=Tanh()):
        super(ScoreNetwork, self).__init__()
        self.query_dim = query_dim
        self.hidden_dim = hidden_dim
        self.query_proj = Linear(query_dim, hidden_dim, bias=True)
        self.non_lin = non_linearity
        self.hidden_to_out_proj = Linear(hidden_dim, 1)

    def forward(self, query, key):
        """
        :param query: [batch_size, query_dim]
        :param key: [batch_size, seq_len, hidden_dim]
        :return: out: [batch_size, seq_len, 1]
        """
        assert key.size(2) == self.hidden_dim
        query = self.query_proj(query)
        # hiddens: [batch_size, seq_len, hidden_dim]
        hidden = self.non_lin(query.unsqueeze(1) + key)
        out = self.hidden_to_out_proj(hidden)
        return out
