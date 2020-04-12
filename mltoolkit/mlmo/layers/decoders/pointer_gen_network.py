from torch.nn import Module
import torch as T


class PointerGenNetwork(Module):
    """Pointer network from See 2017. Specific to one step computations."""

    def __init__(self, gen, copy_gate, ext_vocab_size):
        super(PointerGenNetwork, self).__init__()
        self.ext_vocab_size = ext_vocab_size
        self.gen = gen
        self.copy_gate = copy_gate

    def forward(self, embds, dec_out, att_weights, att_word_ids,
                copy_prob=None):
        """
        :param embds: [batch_size, emb_dim]
                      embeddings of decoder conditioned words.
        :param dec_out: [batch_size, dec_out_dim]
                        concatenated decoder hidden states and context vectors.
        :param att_weights: [batch_size, inp_seq_len]
                            normalized attention scores over inputs.
        :param att_word_ids: [batch_size, inp_seq_len]
        :param copy_prob: scalar value allowing to fix the copy probability.
        :return: word_probs: [batch_size, out_seq_len, ext_vocab_size]
                             word probabilities over the extended vocabulary.
                 prob_ptr: [batch_size, output_seq_len]
        """
        inp_seq_len = att_word_ids.size(1)
        bs = dec_out.size(0)
        device = dec_out.device
        word_probs = T.zeros((bs, self.ext_vocab_size), device=device)

        # project the decoder's output to the vocabulary
        gen_out_probs = self.gen(dec_out)  # [batch_size, vocab_size]
        vocab_size = gen_out_probs.size(-1)

        # distribute probabilities between generator and pointer
        if copy_prob is None:
            # prob_ptr: [batch size, out_seq_len, 1]
            prob_ptr = self.copy_gate(T.cat((dec_out, embds), -1))
            prob_gen = 1. - prob_ptr
        else:
            prob_ptr = T.empty((bs, 1), device=device).fill_(copy_prob)
            prob_gen = 1. - prob_ptr

        # add generator probabilities to output
        word_probs[:, :vocab_size] = prob_gen * gen_out_probs

        if copy_prob is None or copy_prob > 0.:
            att_probs = prob_ptr * att_weights  # [batch_size, inp_seq_len]
            word_probs.scatter_add_(1, att_word_ids, att_probs)

        prob_ptr = prob_ptr.squeeze(-1)

        return word_probs, prob_ptr
