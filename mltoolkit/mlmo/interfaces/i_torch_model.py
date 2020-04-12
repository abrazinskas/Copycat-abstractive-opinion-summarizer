from mltoolkit.mlmo.utils.constants.checkpoint import MODEL_PARAMS, \
    OPTIMIZER_STATE
from mltoolkit.mlmo.interfaces import BaseIModel
from torch.nn import Module
from mltoolkit.mlmo.utils.helpers.loading_and_saving import load_embeddings
from torch.optim import Adam
from logging import getLogger
import torch as T
from torch.nn.utils import clip_grad_norm_
from mltoolkit.mlutils.tools.signature_scraper import repr_func
from mltoolkit.mlutils.helpers.general import select_matching_kwargs
from mltoolkit.mlmo.utils.helpers.pytorch.init import get_init_func
import os

logger = getLogger(os.path.basename(__file__))


class ITorchModel(BaseIModel):
    """
    PyTorch model specific interface. Contains generic methods for training
    and evaluation.
    """

    def __init__(self, model, learning_rate, device='cpu', optimizer=Adam,
                 grads_clip=None, **kwargs):
        """
        :param grads_clip: if float is passed will not allow gradients to exceed
                           a certain threshold. Allows to prevent gradients
                           explosion associated with RNNs.
        """
        if not isinstance(model, Module):
            raise ValueError("Please provide a valid PyTorch model.")
        super(ITorchModel, self).__init__(model, **kwargs)
        self.optimizer = optimizer(self.model.parameters(), lr=learning_rate)
        self.device = device
        self.model.to(device)
        logger.info("Moved the model to: '%s'" % device)
        self.scraper.scrape_obj_vals = False
        self.grads_clip = grads_clip

    def train(self, batch, **kwrgs):
        """
        Performs a training step on a single batch. Returns the internal
        metrics in a dict, such as negative log-likelihood or KL divergence.
        """
        self.model.train()  # setting the model to the train mode
        self.optimizer.zero_grad()  # zero the gradients before backward pass
        kwargs = select_matching_kwargs(self.model.forward, **batch.data)
        kwargs = _move_kwargs_to_device(device=self.device, **kwargs)
        _add_kwargs_to_dict(dct=kwargs, **kwrgs)
        loss, metrs = self.model(**kwargs)
        loss.backward()

        # clips the gradient
        if self.grads_clip is not None:
            clip_grad_norm_(self.model.parameters(), self.grads_clip)

        self.optimizer.step()

        return metrs

    def eval(self, batch, **kwrgs):
        """
        Performs computation of loss and internal metrics on a single batch.
        Same as training, but the model is not updated. Returns the internal
        metrics in a dict.
        """
        self.model.eval()  # setting the model to the test mode
        kwargs = select_matching_kwargs(self.model.forward, **batch.data)
        kwargs = _move_kwargs_to_device(device=self.device, **kwargs)
        _add_kwargs_to_dict(dct=kwargs, **kwrgs)
        with T.no_grad():
            loss, metrs = self.model(**kwargs)
        return metrs

    def save_state(self, file_path, excl_model_params=None):
        model_params = self.model.state_dict()
        optimizer_params = self.optimizer.state_dict()
        if excl_model_params is not None:
            for p in excl_model_params:
                del model_params[p]
        T.save({MODEL_PARAMS: model_params,
                OPTIMIZER_STATE: optimizer_params}, file_path)
        logger.info("Saved the model's and optimizer's state to: '%s'." %
                    file_path)

    def load_state(self, file_path, optimizer_state=False, strict=True):
        checkpoint = T.load(file_path, map_location=self.device)
        self.model.load_state_dict(checkpoint[MODEL_PARAMS], strict=strict)
        logger.info("Loaded the model's state from: '%s'." % file_path)
        if optimizer_state:
            self.optimizer.load_state_dict(checkpoint[OPTIMIZER_STATE])
            logger.info("Loaded the optimizer's state from: '%s'." % file_path)

    def init_weights(self, multi_dim_init_func, single_dim_init_func):
        """Initializes weights using provided functions."""
        logger.info("Initializing multi-dim weights with:"
                    " %s." % repr_func(multi_dim_init_func))
        logger.info("Initializing single-dim weights with:"
                    " %s." % repr_func(single_dim_init_func))
        init = get_init_func(multi_dim_init_func, single_dim_init_func)
        self.model.apply(init)

    def init_embeddings(self, file_path, embds_layer_name, vocab):
        """Sets input and output embedding tensors with pre-trained ones."""
        embs = load_embeddings(file_path, vocab=vocab)
        embd_matr = T.tensor(embs).to(self.device)
        getattr(self.model, embds_layer_name).weight.data = embd_matr

    def __str__(self):
        return str(self.model) + "\n" + str(self.optimizer)


def _move_kwargs_to_device(device, **kwargs):
    for k in kwargs:
        kwargs[k] = kwargs[k].to(device)
    return kwargs


def _add_kwargs_to_dict(dct, **kwargs):
    """Adds new key-value pairs in-place to 'dct'."""
    for k, v in kwargs.items():
        if k in dct:
            raise ValueError("Key '%s' is already present in 'dct'.")
        dct[k] = v
