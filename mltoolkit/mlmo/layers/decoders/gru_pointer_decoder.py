import torch as T
from torch.nn import Module, Linear, GRUCell


class GruPointerDecoder(Module):
    """
    Pointer-generator network based GRU decoder that attends a custom sequence.

    At each time step it run the attention mechanism and computes the context
    vector that together with the decoder's hidden state are passed to the
    pointer-generator network.

    It optionally concatenates the previous context vectors to the word
    embeddings, then passes through a GRU cell. Additional inputs are permitted.
    """

    def __init__(self, input_dim, contxt_dim, hidden_dim, att_module,
                 pointer_gen_module, cat_contx_to_inp=True,
                 pass_extra_feat_to_pg=False, copy_prob=None, **kwargs):
        """
        :param pass_extra_feat_to_pg: whether to pass (concatenated) extra
                                      features to the pointer-generator network
                                      or just embeddings.
        """
        super(GruPointerDecoder, self).__init__()

        self.input_dim = input_dim
        self.contxt_dim = contxt_dim
        self.hidden_dim = hidden_dim
        self.att = att_module
        self.cat_conxt_to_inp = cat_contx_to_inp

        cell_inp_dim = input_dim + contxt_dim if cat_contx_to_inp else input_dim

        self.gru_cell = GRUCell(cell_inp_dim, hidden_dim, **kwargs)
        self.pgn = pointer_gen_module
        self.copy_prob = copy_prob
        self.pass_extra_feat_to_pg = pass_extra_feat_to_pg

    def forward(self, embds, mask, init_hidden, att_keys, att_values,
                att_word_ids, att_mask=None, init_cont=None, extra_feat=None,
                **att_kwargs):
        """
        Performs forward pass given input and optional values to attend. If
        attention values are not provided, the corresponding output vector's
        part will be filled with zeroes.

        :param embds: [batch_size, out_seq_len, input_dim]
                      embedded representations of each output sequence time-step.
        :param mask: [batch_size, out_seq_len]
        :param init_hidden: [batch_size, hidden_dim]
        :param att_keys: [batch_size, inp_seq_len, att_hidden_dim]
        :param att_word_ids: [batch_size, inp_seq_len]
                             word ids corresponding to input sequences.
        :param att_values: [batch_size, inp_seq_len, value_dim]
        :param att_mask: [batch_size, inp_seq_len]
        :param init_cont: [batch_size, value_dim]
                          initial context vector.
        :param extra_feat: [batch_size, out_seq_len, dim]
                           extra representations or features that should be
                           concatenated to the embeddings.
        :param att_kwargs: additional parameters that should be passed to the
                           forward method of the attention module.
        :return: outs: [batch_size, out_seq_len, *]
                       hidden states concatenated with contxt_vecs
                 prev_hidden: [batch_size, hidden_dim]
                              last hidden layer.
                 att_weights: [batch_size, out_seq_len, inp_seq_len]
                              normalized scores of where the model was attending
                              over the 'out_seq_len' steps decoding.
                 prev_cont: [batch_size, value_dim]
                            last context vector produced by the attention.
        """
        assert len(embds.shape) == 3
        assert len(init_hidden.shape) == 2
        assert len(att_values.shape) == 3
        assert len(att_mask.shape) == 2
        bs = embds.size(0)
        out_seq_len = embds.size(1)
        inp_seq_len = att_values.size(1)
        device = embds.device
        init_hidden = init_hidden

        # collectors
        word_probs = T.empty((bs, out_seq_len, self.pgn.ext_vocab_size),
                             dtype=T.float32, device=device)
        copy_probs = T.empty((bs, out_seq_len), device=device)
        att_weights = T.empty((bs, out_seq_len, inp_seq_len), dtype=T.float32,
                              device=device)
        if init_cont is None:
            init_cont = T.zeros((bs, self.contxt_dim), device=device)

        prev_hidden = init_hidden
        prev_cont = init_cont

        mask = mask.unsqueeze(-1)

        for indx in range(out_seq_len):
            _x = embds[:, indx]
            _m = mask[:, indx]

            gru_inp = _x if extra_feat is None \
                else T.cat((_x, extra_feat[:, indx]), dim=-1)

            hidden, cont, att_wts = self.step(gru_inp, _m,
                                              prev_hidden=prev_hidden,
                                              prev_cont=prev_cont,
                                              att_keys=att_keys,
                                              att_values=att_values,
                                              att_mask=att_mask,
                                              **att_kwargs)

            out = T.cat((hidden, cont), dim=1)

            if self.pass_extra_feat_to_pg and extra_feat is not None:
                pgn_inp = T.cat((_x, extra_feat[:, indx]), dim=-1)
            else:
                pgn_inp = _x

            # running the pointer-generator network
            w_probs, c_probs = self.pgn(embds=pgn_inp, dec_out=out,
                                        att_weights=att_wts,
                                        att_word_ids=att_word_ids,
                                        copy_prob=self.copy_prob)

            att_weights[:, indx] = att_wts
            word_probs[:, indx] = w_probs
            copy_probs[:, indx] = c_probs

            prev_hidden = hidden
            prev_cont = cont

        return word_probs, copy_probs, att_weights, prev_hidden, prev_cont

    def step(self, emb, m, prev_hidden, prev_cont, att_keys, att_values,
             att_mask, **att_kwargs):
        """
        Performs a decoding step by running the GRU cell.

        :param emb: [batch_size, emb_dim]
                    word embeddings.
        :param m: [batch_size]
                  mask for the current time-step.
        :param prev_cont: [batch_size, conxt_dim]
        :param att_keys: [batch_size, seq_len, dim]
        :param att_values: [batch_size, seq_len, dim]
        :param att_mask: [batch_size]
        """
        if self.cat_conxt_to_inp:
            inp = T.cat((emb, prev_cont), dim=1)
        else:
            inp = emb

        curr_hidden = self.gru_cell(inp, prev_hidden)

        curr_cont, att_wts = self.att(query=curr_hidden, key=att_keys,
                                      value=att_values, mask=att_mask,
                                      **att_kwargs)

        # copying the hidden layer and context vectors for masked seqs
        new_hidden = m * curr_hidden + (1. - m) * prev_hidden
        new_cont = m * curr_cont + (1. - m) * prev_cont

        return new_hidden, new_cont, att_wts
