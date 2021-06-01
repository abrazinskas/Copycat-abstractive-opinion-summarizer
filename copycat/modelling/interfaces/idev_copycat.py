from logging import getLogger
from mltoolkit.mldp.utils.constants.vocabulary import PAD, UNK, START, END
import os
from time import time
from mltoolkit.mlutils.helpers.formatting.general import metrics_to_str
from copycat.utils.fields import ModelF
from mltoolkit.mlmo.interfaces import BaseIDev
from collections import OrderedDict
import numpy as np
from mltoolkit.mlutils.helpers.paths_and_files import safe_mkfdir
from mltoolkit.mldp.steps.transformers.nlp import VocabMapper
from mltoolkit.mldp.utils.tools import DataChunk
from mltoolkit.mlmo.utils.helpers.pytorch.data import convert_tensors_to_numpy
from mltoolkit.mlmo.utils.helpers.analytics import ngram_seq_analysis
from mltoolkit.mldp.utils.helpers.dc import concat_chunks
from mltoolkit.mlutils.helpers.formatting.seq import format_seqs, \
    conv_seqs_to_sents
from mltoolkit.mlutils.helpers.formatting.general import format_big_box
from copycat.utils.helpers.data import group_vals_by_keys
from mltoolkit.mlmo.utils.tools.annealing import KlMonAnnealing, KlCycAnnealing
import warnings
from torch.cuda import empty_cache

logger_name = os.path.basename(__file__)
logger = getLogger(logger_name)


class IDevCopyCat(BaseIDev):
    """Development interface for CopyCat."""

    def __init__(self, word_vocab, c_kl_ann, z_kl_ann,
                 tok_func, detok_func, sent_split_func, **kwargs):
        assert isinstance(c_kl_ann, (KlMonAnnealing, KlCycAnnealing))
        assert isinstance(z_kl_ann, (KlMonAnnealing, KlCycAnnealing))
        super(IDevCopyCat, self).__init__(**kwargs)
        self.c_kl_ann = c_kl_ann
        self.z_kl_ann = z_kl_ann
        self.word_vocab = word_vocab
        self.tok_func = tok_func
        self.detok_func = detok_func
        self.sent_split_func = sent_split_func

        self.scraper.scrape_obj_vals = False

    def train(self, data_source, logging_steps=10, **kwargs):
        """
        Performs a single epoch training on the passed `data_source`.

        :param data_source: self-explanatory.
        :param logging_steps: how often to log training produced batch metrics.
        """
        empty_cache()
        logger.info("Training data source: %s" % data_source)
        total_batches = 0
        total_revs = 0
        start = time()

        data_chunk_iter = self.train_data_pipeline.iter(**data_source)

        for indx, batch in enumerate(data_chunk_iter, 1):
            c_lambd = self.c_kl_ann(increment_indx=True)
            z_lambd = self.z_kl_ann(increment_indx=True)
            metrics = self.imodel.train(batch, c_lambd=c_lambd, z_lambd=z_lambd)
            total_revs += len(batch[ModelF.REV])
            if indx % logging_steps == 0:
                mess = metrics_to_str(metrics, prefix="Chunk # %d" % indx)
                logger.info(mess)
            total_batches += 1

        logger.info("Epoch training time elapsed: %.2f (s)." % (time() - start))
        logger.info("Total reviews: %d." % total_revs)

    def eval_on_intern_metrs(self, data_source, **kwargs):
        """Computes the average over data points metrics."""
        empty_cache()
        logger.info("Evaluation data source: %s" % data_source)
        total_metrs = OrderedDict()
        total_dp = 0
        total_batches = 0
        total_revs = 0
        start = time()
        for batch in self.val_data_pipeline.iter(**data_source):
            metrs = self.imodel.eval(batch=batch, **kwargs)
            total_revs += len(batch[ModelF.REV])
            for k, v in metrs.items():
                if k not in total_metrs:
                    total_metrs[k] = 0.
                total_metrs[k] += v * len(batch)  # rescaling back
            total_dp += len(batch)
            total_batches += 1

        logger.info("Evaluation time elapsed: %.2f (s)." % (time() - start))
        logger.info("Total reviews: %d." % total_revs)

        # compute the actual average over data-points
        f_res = OrderedDict()
        for k, v in total_metrs.items():
            f_res[k] = v / float(total_dp)

        return f_res

    def gen_and_save_summs(self, data_source, output_file_path):
        """
        Generates summaries by running the model and writes them along with other
        attributes to a json file.

        :param data_source: self-explanatory.
        :param output_file_path: self-explanatory.
        """
        safe_mkfdir(output_file_path)
        start_id = self.word_vocab[START].id
        end_id = self.word_vocab[END].id
        pad_id = self.word_vocab[PAD].id
        output_file = open(output_file_path, encoding='utf-8', mode='w')
        vocab_mapper = VocabMapper({ModelF.REV: self.word_vocab,
                                    ModelF.GEN_SUMM: self.word_vocab,
                                    ModelF.GEN_REV: self.word_vocab},
                                   symbols_attr='token')
        chunk_coll = []

        for i, dc in enumerate(self.val_data_pipeline.iter(**data_source), 1):
            gen_revs, _, gen_summ, _ = self.imodel.predict(dc)

            # converting to the data-chunk to use the internal writing
            # mechanism
            new_dc = DataChunk()
            for fn in [ModelF.SUMM_CAT, ModelF.SUMM_GROUP_ID, ModelF.REV,
                       ModelF.GROUP_ID]:
                new_dc[fn] = dc[fn]
            new_dc[ModelF.GEN_REV] = gen_revs
            new_dc[ModelF.GEN_SUMM] = gen_summ

            seq_fnames = [ModelF.GEN_SUMM, ModelF.GEN_REV, ModelF.REV]
            # converting PyTorch tensors to numpy arrays if present
            new_dc = convert_tensors_to_numpy(new_dc)
            for fn in seq_fnames:
                new_dc[fn] = format_seqs(new_dc[fn], start_id=start_id,
                                         end_id=end_id, pad_id=pad_id)
            new_dc = vocab_mapper(new_dc)

            # convert all seqs to strings
            for fn in seq_fnames:
                new_dc[fn] = conv_seqs_to_sents(new_dc[fn])

            # group by product ids
            indxs = group_vals_by_keys(range(len(new_dc[ModelF.REV])),
                                       new_dc[ModelF.GROUP_ID]).values()

            for fn in [ModelF.GEN_REV, ModelF.REV]:
                new_dc[fn] = self._group_by_prods(indxs, new_dc[fn])

            del new_dc[ModelF.GROUP_ID]

            chunk_coll.append(new_dc)

        output_chunk = concat_chunks(*chunk_coll)

        output_chunk.to_json(f=output_file,
                             grouping_fnames=[ModelF.SUMM_CAT,
                                              ModelF.SUMM_GROUP_ID])

        logger.info("Generated summaries and saved to: '%s'."
                    "" % output_file_path)

        # analytics for repetitions checking
        # because gen summs contain list of strings I need to merge them
        # together before running analytics
        all_gen_summ_strs = [" ".join(sents) for sents in
                             output_chunk[ModelF.GEN_SUMM]]

        an_metrics = ngram_seq_analysis(all_gen_summ_strs,
                                        tokenizer=self.tok_func,
                                        sent_splitter=self.sent_split_func,
                                        n_grams_to_comp=(2, 3, 4))

        logger.info("Ran analytics of generated summaries.")
        metrs_str = " ".join(
            ["%s: %.3f" % (k, v) for k, v in an_metrics])
        logger.info(metrs_str)

    def save_setup_str(self, dir_path, exper_descr=None):
        """
        Logs/saves the setup of the experiment, namely 3 main components:
        1. Logs the experiment's description (if `exper_descr` is provided).
        2. Train and val data pipelines' and vocab' blueprint, saved
            to dp_vocabs.txt.
        3. Model's blueprint/summary saved to model.txt.
        """
        logger.info("Experiment's output will be saved to: '%s'." % dir_path)
        # 1. experiment
        if exper_descr:
            form_exp = format_big_box(exper_descr)
            logger.info(form_exp)

        # 2. data pipeline + vocabs
        dp_fp = os.path.join(dir_path, 'dp_vocabs.txt')
        safe_mkfdir(dp_fp)
        try:
            with open(dp_fp, 'w') as f:
                f.write(str(self.word_vocab))
                f.write('===========================')
                f.write(str(self.train_data_pipeline))
                f.write('===========================')
                f.write(str(self.val_data_pipeline))

        except Exception:
            os.remove(dp_fp)
            warnings.warn(
                "Could not get the str of the dev_data_pipeline's setup.")

        # 3. model and its interface
        m_fp = os.path.join(dir_path, 'model.txt')
        try:
            with open(m_fp, 'w') as f:
                f.write(str(self.imodel))
        except Exception:
            os.remove(m_fp)
            warnings.warn("Could not get the str of the model's setup.")

    def summ_generator(self, batch, summ_post_proc=None):
        """
        Inputs a batch and returns a list of tokens of generated summaries.

        :return: list of strings.
        """
        start_id = self.word_vocab[START].id
        end_id = self.word_vocab[END].id
        pad_id = self.word_vocab[PAD].id

        summs_word_ids = self.imodel.generate_summaries(batch)

        summs_word_ids = format_seqs(summs_word_ids, start_id=start_id,
                                     end_id=end_id, pad_id=pad_id,
                                     append_end=False)
        summs_tokens = self._conv_seq_ids_to_tokens(summs_word_ids)

        summs_strs = [" ".join(summ) for summ in summs_tokens]

        if summ_post_proc:
            summs_strs = [summ_post_proc(s_str) for s_str in summs_strs]

        return summs_strs

    def _conv_seq_ids_to_tokens(self, seqs, excl=None):
        coll = []
        excl = [] if excl is None else excl
        for seq in seqs:
            tokens = [self.word_vocab[id].token for id in seq if
                      id not in excl]
            coll.append(tokens)
        return coll

    @staticmethod
    def _group_by_prods(lists_of_indxs, fvals):
        tmp = np.zeros(len(lists_of_indxs), dtype='object')
        for i, indxs in enumerate(lists_of_indxs):
            tmp[i] = list(fvals[indxs])
        return tmp

    def format_revs(self, revs):
        """
        Formats reviews to make them visualizible. Used by the summary evaluator.

        :param revs: batch of reviews. Data-type depends on the data-pipeline.
        :return: list of strings
        """
        start_id = self.word_vocab[START].id
        end_id = self.word_vocab[END].id
        pad_id = self.word_vocab[PAD].id
        revs = revs.to('cpu').numpy()
        revs = format_seqs(revs, start_id=start_id, end_id=end_id,
                           pad_id=pad_id, append_end=False)
        revs = self._conv_seq_ids_to_tokens(revs)
        revs = [self.detok_func(rev) for rev in revs]

        return revs
