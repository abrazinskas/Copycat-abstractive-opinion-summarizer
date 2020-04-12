from mltoolkit.mlmo.layers import Ffnn, MuSigmaFfnn, Attention, OutEmbds
from mltoolkit.mlmo.layers.encoders import GruEncoder
from mltoolkit.mlmo.layers.decoders import PointerGenNetwork, GruPointerDecoder
from torch.nn import Module, Embedding, Tanh, Sequential, Softmax, \
    Sigmoid, Parameter, Linear
from mltoolkit.mlmo.utils.helpers.pytorch.computation import comp_seq_log_prob, \
    re_parameterize, kld_gauss, kld_normal, masked_softmax
from collections import OrderedDict
import torch as T
from copycat.utils.helpers.modelling import group_att_over_input
from mltoolkit.mlmo.utils.tools import DecState

EPS = 1e-12


class CopyCat(Module):
    """
    CopyCat summarizer variational summarizer based on hierarchical latent
    representations of data. Specifically, it represents both review groups
    and reviews as separate latent code types.
    """

    def __init__(self, vocab_size, ext_vocab_size, emb_dim, enc_hidden_dim,
                 c_dim, z_dim, att_hidden_dim, states_sc_hidden=150,
                 cgate_hidden_dim=None):
        """
        :param vocab_size: the number of words by the generator.
        :param ext_vocab_size: the extended number of words that is accessible
            by the copy mechanism.
        :param emb_dim: the number of dimensions in word embeddings.
        :param enc_hidden_dim: GRU encoder's hidden dimension.
        :param c_dim: dimension of the group representation.
        :param z_dim: dimension of the review representation.
        :param att_hidden_dim: hidden dimension of hidden dimension of
            feed-forward networks.
        :param states_sc_hidden: hidden dimension of the score function used
            for production of the group representation c.
        :param cgate_hidden_dim: copy gate hidden dimension.
        """
        assert vocab_size <= ext_vocab_size

        super(CopyCat, self).__init__()
        self._embds = Embedding(ext_vocab_size, emb_dim)

        self._encoder = GruEncoder(input_dim=emb_dim, hidden_dim=enc_hidden_dim)

        # POINTER-GENERATOR NETWORK #

        #   generation network - computes generation distribution over words
        dec_inp_dim = z_dim + enc_hidden_dim
        gen_network = Sequential()
        gen_network.add_module("lin_proj", Linear(dec_inp_dim, emb_dim))
        gen_network.add_module("out_embds", OutEmbds(self._embds, vocab_size))
        gen_network.add_module("softmax", Softmax(dim=-1))

        #   copy gate - computes probability of copying a word
        c_ffnn = Ffnn(z_dim + enc_hidden_dim + emb_dim,
                      hidden_dim=cgate_hidden_dim,
                      non_linearity=Tanh(), output_dim=1)
        copy_gate = Sequential()
        copy_gate.add_module("ffnn", c_ffnn)
        copy_gate.add_module("sigmoid", Sigmoid())

        pgn = PointerGenNetwork(gen=gen_network, copy_gate=copy_gate,
                                ext_vocab_size=ext_vocab_size)

        # COMPLETE DECODER (GRU + ATTENTION + COPY-GEN) #

        #   attention for decoder over encoder's hidden states
        dec_att = Attention(query_dim=z_dim, hidden_dim=att_hidden_dim,
                            value_dim=enc_hidden_dim,
                            non_linearity=Tanh())
        self._keys_creator = dec_att.create_keys
        self._decoder = GruPointerDecoder(input_dim=emb_dim + z_dim,
                                          hidden_dim=z_dim,
                                          contxt_dim=enc_hidden_dim,
                                          att_module=dec_att,
                                          pointer_gen_module=pgn,
                                          cat_contx_to_inp=True,
                                          pass_extra_feat_to_pg=False)

        # NETWORKS THAT PRODUCES LATENT VARIABLES' MUS AND SIGMAS #

        # z inference network
        self._z_inf_network = MuSigmaFfnn(input_dim=enc_hidden_dim + c_dim,
                                          non_linearity=Tanh(),
                                          output_dim=z_dim)
        # z prior network
        self._z_prior_network = MuSigmaFfnn(input_dim=c_dim, output_dim=z_dim)

        # c inference network
        #   scores each state to obtain group context vector
        states_dim = enc_hidden_dim + emb_dim
        self._c_states_scoring = Ffnn(input_dim=states_dim,
                                      hidden_dim=states_sc_hidden,
                                      output_dim=1, non_linearity=Tanh())

        self._c_inf_network = MuSigmaFfnn(input_dim=states_dim,
                                          output_dim=c_dim)

    def forward(self, rev, rev_len, rev_mask,
                group_rev_indxs, group_rev_indxs_mask,
                rev_to_group_indx, other_rev_indxs, other_rev_indxs_mask,
                other_rev_comp_states, other_rev_comp_states_mask,
                c_lambd=0., z_lambd=0.):
        """
        :param rev: review word ids.
            [batch_size, rev_seq_len]
        :param rev_len: review lengths.
            [batch_size]
        :param rev_mask: float mask where 0. is set to padded words.
            [batch_size, rev_seq_len]
        :param rev_to_group_indx: mapping from reviews to their corresponding
            groups.
            [batch_size]
        :param group_rev_indxs: indxs of reviews that belong to same groups.
            [group_count, max_rev_count]
        :param group_rev_indxs_mask: float mask where 0. is set to padded
            review indxs.
            [group_count, max_rev_count]
        :param other_rev_indxs: indxs of leave-one-out reviews.
            [batch_size, max_rev_count]
        :param other_rev_indxs_mask: float mask for leave-one-out reviews.
        :param other_rev_comp_states: indxs of (hidden) states of leave-one-out
            reviews. Used as an optimization to avoid attending over padded
            positions.
            [batch_size, cat_rev_len]
        :param other_rev_comp_states_mask: masking of states for leave-one-out
            reviews.
            [batch_size, cat_rev_len]
        :param c_lambd: annealing constant for c representations.
        :param z_lambd: annealing constant for z representations.

        :return loss: scalar loss corresponding to the mean ELBO over batches.
        :return metrs: additional statistics that are used for analytics and
            debugging.
        """
        bs = rev.size(0)
        device = rev.device
        group_count = group_rev_indxs.size(0)
        loss = 0.
        metrs = OrderedDict()

        rev_word_embds = self._embds(rev)
        rev_encs, rev_hiddens = self.encode(rev_word_embds, rev_len)

        att_keys = self.create_att_keys(rev_hiddens)
        contxt_states = self.get_contxt_states(rev_hiddens, rev_word_embds)

        # running the c inference network for the whole group
        c_mu_q, \
        c_sigma_q, \
        scor_wts = self.get_q_c_mu_sigma(contxt_states, rev_mask,
                                         group_rev_indxs, group_rev_indxs_mask)
        # c_mu_q: [group_count, context_dim]
        # c_sigma_q: [group_count, context_dim]
        c = re_parameterize(c_mu_q, c_sigma_q)

        # running the z inference network for each review
        z_mu_q, z_sigma_q = self.get_q_z_mu_sigma(rev_encs,
                                                  c=c[rev_to_group_indx])
        z = re_parameterize(z_mu_q, z_sigma_q)

        # PERFORMING REVIEWS RECONSTRUCTION #

        rev_att_keys, \
        rev_att_vals, \
        rev_att_mask = group_att_over_input(inp_att_keys=att_keys,
                                            inp_att_vals=rev_hiddens,
                                            inp_att_mask=rev_mask,
                                            att_indxs=other_rev_indxs,
                                            att_indxs_mask=other_rev_indxs_mask)

        rev_att_word_ids = rev[other_rev_indxs].view(bs, -1)

        # optimizing the attention targets by making more compact tensors
        # with less padded entries
        sel = T.arange(bs, device=device).unsqueeze(-1)
        rev_att_keys = rev_att_keys[sel, other_rev_comp_states]
        rev_att_vals = rev_att_vals[sel, other_rev_comp_states]
        rev_att_mask = other_rev_comp_states_mask
        rev_att_word_ids = rev_att_word_ids[sel, other_rev_comp_states]

        # creating an extra feature that is passe
        extra_feat = z.unsqueeze(1).repeat(1, rev_word_embds.size(1), 1)

        log_probs, rev_att_wts, \
        hidden, cont, \
        copy_probs = self._decode(embds=rev_word_embds, mask=rev_mask,
                                  extra_feat=extra_feat,
                                  hidden=z, att_keys=rev_att_keys,
                                  att_values=rev_att_vals,
                                  att_mask=rev_att_mask,
                                  att_word_ids=rev_att_word_ids)
        rec_term = comp_seq_log_prob(log_probs[:, :-1], seqs=rev[:, 1:],
                                     seqs_mask=rev_mask[:, 1:])
        avg_rec_term = rec_term.mean(dim=0)

        loss += - avg_rec_term

        # KULLBACK-LEIBLER TERMS #

        # c kld
        c_kl_term = kld_normal(c_mu_q, c_sigma_q)
        # notice that the below kl term is divided by the number of
        # reviews (data-units) to permit proper scaling of loss components
        summed_c_kl_term = c_kl_term.sum()
        avg_c_kl_term = summed_c_kl_term / bs
        loss += c_lambd * avg_c_kl_term

        # z kld
        # running the prior network
        z_mu_p, z_sigma_p = self.get_p_z_mu_sigma(c=c)
        # computing the actual term
        z_kl_term = kld_gauss(z_mu_q, z_sigma_q, z_mu_p[rev_to_group_indx],
                              z_sigma_p[rev_to_group_indx], eps=EPS)
        avg_z_kl_term = z_kl_term.mean(dim=0)
        loss += z_lambd * avg_z_kl_term

        log_var_p_z = T.log(z_sigma_p).sum(-1)

        rev_att_wts = rev_att_wts[:, :-1] * rev_mask[:, 1:].unsqueeze(
            -1)
        copy_probs = copy_probs[:, :-1] * rev_mask[:, 1:]

        # computing maximums over attention weights
        rev_att_max = rev_att_wts.max(-1)[0]

        # computing maximums over steps of copy probs
        max_copy_probs = copy_probs.max(-1)[0]

        log_var_q_z = T.log(z_sigma_q).sum(-1)
        log_var_q_c = T.log(c_sigma_q).sum(-1)

        # averaging over batches different statistics for logging
        avg_att_max = rev_att_max.mean()
        avg_max_copy_prob = max_copy_probs.mean(0)
        avg_copy_prob = (copy_probs.sum(-1) / rev_len.float()).mean(0)
        avg_first_scor_wts = scor_wts[:, 0].mean(0)

        metrs['avg_lb'] = (avg_rec_term - avg_c_kl_term - avg_z_kl_term).item()
        metrs['avg_rec'] = avg_rec_term.item()
        metrs['avg_c_kl'] = (summed_c_kl_term / group_count).item()
        metrs['avg_z_kl'] = avg_z_kl_term.item()

        metrs['avg_log_p_z_var'] = log_var_p_z.mean(0).item()
        metrs['avg_log_q_z_var'] = log_var_q_z.mean(0).item()
        metrs['avg_log_q_c_var'] = log_var_q_c.mean(0).item()
        metrs['avg_att_max'] = avg_att_max.item()
        metrs['avg_max_copy_prob'] = avg_max_copy_prob.item()
        metrs['avg_copy_prob'] = avg_copy_prob.item()

        metrs['c_lambd'] = c_lambd
        metrs['z_lambd'] = z_lambd

        return loss, metrs

    def get_p_z_mu_sigma(self, c):
        """Computes z prior's parameters (mu, sigma)."""
        mu_p, sigma_p = self._z_prior_network(c)
        return mu_p, sigma_p

    def get_q_z_mu_sigma(self, encs, c):
        """
        Runs the inference network and computes mu and sigmas for review latent
        codes (z).
        """
        inp = T.cat((encs, c), dim=-1)
        mu_q, sigma_q = self._z_inf_network(inp)
        return mu_q, sigma_q

    def get_q_c_mu_sigma(self, states, states_mask, group_indxs,
                         group_indxs_mask):
        """Computes c approximate posterior's parameters (mu, sigma).

        :param states: [batch_size, seq_len1, dim]
                        representations of review steps, e.g. hidden + embd.
        :param states_mask: [batch_size, seq_len1]
        :param group_indxs: [batch_size2, seq_len2]
                      indxs of reviews belonging to the same product
        :param group_indxs_mask: [batch_size2, seq_len2]
        """
        grouped_states, \
        grouped_mask = group_att_over_input(inp_att_vals=states,
                                            inp_att_mask=states_mask,
                                            att_indxs=group_indxs,
                                            att_indxs_mask=group_indxs_mask)
        ws_state, \
        score_weights = self._compute_ws_state(states=grouped_states,
                                               states_mask=grouped_mask)
        mu, sigma = self._c_inf_network(ws_state)

        return mu, sigma, score_weights

    def encode(self, embds, mask_or_lens):
        return self._encoder(embds, mask_or_lens)

    def decode_beam(self, seqs, hidden, att_word_ids, init_z=None, **kwargs):
        """Function to be used in the beam search process.

        :param seqs: [batch_size, 1]
        :param hidden: [batch_size, hidden_dim]
        :param att_word_ids: [batch_size, cat_rev_len]
        :param init_z: [batch_size, z_dim]
        """
        embds = self._embds(seqs)
        mask = T.ones_like(seqs, dtype=T.float32)

        if init_z is None:
            init_z = hidden

        word_log_probs, att_wts, \
        hidden, cont, ptr_probs = self._decode(embds=embds, mask=mask,
                                               extra_feat=init_z.unsqueeze(1),
                                               hidden=hidden,
                                               att_word_ids=att_word_ids,
                                               **kwargs)
        out = DecState(word_scores=word_log_probs,
                       rec_vals={"hidden": hidden, "cont": cont,
                                 "init_z": init_z},
                       coll_vals={'copy_probs': ptr_probs.squeeze(-1),
                                  'att_wts': att_wts.squeeze(1),
                                  "att_word_ids": att_word_ids})
        return out

    def _decode(self, embds, mask, hidden, cont=None, eps=EPS, **dec_kwargs):
        """Teacher forcing decoding of sequences.

        :param embds: [batch_size, seq_len, dim]
        :param mask: [batch_size, seq_len]
        :param att_word_ids: [batch_size, inp_seq_len]
        :param copy_prob: a fixed probability of copying words.
        :param hidden: [batch_size, hidden_dim]
        """
        word_probs, copy_probs, \
        att_weights, prev_hidden, \
        prev_cont = self._decoder(embds, mask, init_hidden=hidden,
                                  init_cont=cont, **dec_kwargs)

        word_log_probs = T.log(word_probs + eps)

        return word_log_probs, att_weights, prev_hidden, prev_cont, copy_probs

    def _compute_ws_state(self, states, states_mask):
        """Computes weighted state by scoring each state."""
        state_scores = self._c_states_scoring(states)
        grouped_state_weights = masked_softmax(state_scores, states_mask, dim=1)
        group_context = (states * grouped_state_weights).sum(dim=1)
        return group_context, grouped_state_weights

    def create_att_keys(self, hiddens):
        """
        Creates the attention keys that are used in the decoder.
        Performs projection that is required by the attention mechanism, allows
        for a speed-up by not performing this operation every time attention is
         called (at each decoding step).

        :param hiddens: [batch_size, seq_len, hidden_dim]
        """
        att_keys = self._keys_creator(hiddens)
        return att_keys

    def get_contxt_states(self, hiddens, embds):
        return T.cat((hiddens, embds), dim=-1)
