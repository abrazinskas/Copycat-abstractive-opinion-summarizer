from mltoolkit.mlmo.utils.tools import BaseHP


class ModelHP(BaseHP):
    """
    Contains hyper-parameters of the actual model.
    Please see `CopyCat` for more details on hyper-parameters.
    """

    def __init__(self):
        super(ModelHP, self).__init__()
        self.vocab_size = 50000
        self.ext_vocab_size = 80000
        self.emb_dim = 200
        self.enc_hidden_dim = 600
        self.c_dim = 600
        self.z_dim = 600
        self.states_sc_hidden = 300
        self.att_hidden_dim = 200
        self.cgate_hidden_dim = 100
