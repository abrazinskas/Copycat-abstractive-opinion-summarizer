from copycat.utils.fields import ModelF, OutputF
from copycat.eval.metrics import GoogleRouge as Rouge
from mltoolkit.mlutils.helpers.formatting.general import metrics_to_str
from logging import getLogger
from mltoolkit.mlutils.helpers.paths_and_files import safe_mkfdir
from mltoolkit.mldp.utils.tools import DataChunk
import codecs
import numpy as np
import os

logger_name = os.path.basename(__file__)
logger = getLogger(logger_name)


class SummEvalProc(object):
    """
    Performs evaluation on summaries based on Rouge.
    Writes the results in terms of true and generated
    summaries, and their associated ROUGE scores to a file.
    """

    def __init__(self, data_pipeline, summs_gen_func, revs_formatter_func,
                 sent_splitter=None, avg_rouge=True, analytics_func=None):
        """
        :param data_pipeline: self-explanatory.
        :param summs_gen_func: takes a batch and returns list of string
                               summaries.
        :param revs_formatter_func: takes reviews (field) and formats it
                                    by outputting list of strings.
        :param avg_rouge: whether to compute avg multi-reference ROUGE or max.
        :param analytics_func: if provided will pass the generated summaries
                               though the function.
        """
        self.data_pipeline = data_pipeline
        self.summs_gen_func = summs_gen_func
        self.revs_formatter_func = revs_formatter_func
        self.sent_splitter = sent_splitter
        self.avg_rouge = avg_rouge
        self.analytics_func = analytics_func

    def eval(self, data_source, output_file_path=None):
        """
        Assumes that batches contain SUMMS that are lists of sublists,
        where is sublist contain a fixed number of summary strings. I.e.
        summaries should not be tokenized.

        :param data_source:
        :param output_file_path:
        """
        output_dc = DataChunk(**{OutputF.GOLD_SUMMS: [], OutputF.GEN_SUMM: [],
                                 OutputF.GROUP_ID: [], OutputF.CAT: [],
                                 OutputF.ROUGE: [], OutputF.REV: []})
        rouge_evaluator = Rouge()
        skipped_summs = 0

        for batch in self.data_pipeline.iter(**data_source):
            # notice that each product has K true summaries created by
            # annotators
            true_summs = batch[ModelF.SUMMS]
            prod_ids = batch[ModelF.SUMM_GROUP_ID]
            cats = batch[ModelF.SUMM_CAT]

            # getting group reviews that were used as input to produce summaries
            inp_revs = self.revs_formatter_func(batch[ModelF.REV])
            group_rev_indxs = batch[ModelF.GROUP_REV_INDXS]
            group_rev_indxs_mask = batch[ModelF.GROUP_REV_INDXS_MASK]
            group_revs = get_group_reviews(inp_revs, group_rev_indxs,
                                           group_rev_indxs_mask)

            gen_summs = self.summs_gen_func(batch)

            assert (len(true_summs) == len(gen_summs))

            # accumulating ROUGE statistics
            res = []
            for gen_summ, _true_summs in zip(gen_summs, true_summs):

                if len(gen_summ) == 0:
                    skipped_summs += 1
                    res.append(None)
                    continue

                # extra [] wrapping is needed as the accum method is batch based
                r_avg, _, r_max, _ = rouge_evaluator.accum(
                    hypotheses=[gen_summ],
                    references=[_true_summs])
                if self.avg_rouge:
                    curr_rouge = r_avg
                else:
                    curr_rouge = r_max
                res.append(curr_rouge)

            # splitting by the sentence for better visualization
            if self.sent_splitter:
                group_revs = self.split_group_seqs_by_sents(group_revs)
                true_summs = self.split_group_seqs_by_sents(true_summs)
                gen_summs = [self.sent_splitter(summ) for summ in gen_summs]

            # storing the output batch for later dumping
            output_dc[OutputF.GOLD_SUMMS] += true_summs
            output_dc[OutputF.GEN_SUMM] += gen_summs
            output_dc[OutputF.REV] += group_revs
            output_dc[OutputF.CAT] += list(cats)
            output_dc[OutputF.GROUP_ID] += list(prod_ids)
            output_dc[OutputF.ROUGE] += res

        # running analytics
        if self.analytics_func:
            if self.sent_splitter:
                # performing a preliminary merge of sentences
                summs_to_analyze = [" ".join(sents) for sents in
                                    output_dc[ModelF.GEN_SUMM]]
            else:
                summs_to_analyze = output_dc[ModelF.GEN_SUMM]
            res = self.analytics_func(summs_to_analyze)
            logger.info("Ran analytics of generated summaries.")
            logger.info(" ".join("%s: %.3f" % (k, v) for k, v in res))

        final_metrs = rouge_evaluator.aggr(avg=self.avg_rouge)
        logger.info("ROUGE scores (avg_rouge=%s): " % self.avg_rouge)
        for k, v in final_metrs.items():
            logger.info("%s based avg. %s." % (k, metrics_to_str(v)))

        # this is a safe way to make proper arrays
        for k in output_dc:
            l = len(output_dc[k])
            cont = np.zeros(l, dtype='object')
            for indx in range(l):
                cont[indx] = output_dc[k][indx]
            output_dc[k] = cont

        if output_file_path:
            gr_fields = [OutputF.CAT, OutputF.GROUP_ID]
            safe_mkfdir(output_file_path)
            output_file = codecs.open(output_file_path, 'w')
            output_dc.to_json(f=output_file, grouping_fnames=gr_fields)
            logger.info("Wrote the eval output to: "
                        "'%s'." % output_file_path)
        logger.info("Not generated %d summaries." % skipped_summs)

    def split_group_seqs_by_sents(self, group_seqs):
        coll = []
        for _group in group_seqs:
            tmp = []
            for rev in _group:
                tmp.append(self.sent_splitter(rev))
            coll.append(tmp)
        return coll


def get_group_reviews(revs, group_rev_indxs, group_rev_indxs_mask):
    """For each group returns its input reviews considering masking."""
    coll = []
    for indxs, mask in zip(group_rev_indxs, group_rev_indxs_mask):
        coll.append([revs[indx] for indx, m in zip(indxs, mask) if m != 0])
    return coll
