# contains functions associated with run scripts #
import os
from logging import getLogger
from mltoolkit.mlutils.helpers.paths_and_files import get_file_name, comb_paths
from copycat.eval.procedures import SummEvalProc

logger_name = os.path.basename(__file__)
logger = getLogger(logger_name)


def gen_seqs(output_folder, gen_func, data_sources):
    """
    Generates the output (reconstructed sequences and summaries) based on the
    provided data-sources.
    """
    if not isinstance(data_sources, list):
        data_sources = [data_sources]
    logger.info("Performing generation of sequences/summaries.")
    for data_source in data_sources:
        fn = get_file_name(data_source['data_path'])
        if fn == '':
            _, fn = os.path.split(os.path.dirname(data_source['data_path']))
        ofp = comb_paths(output_folder, "%s.json" % fn)
        gen_func(data_source, ofp)


def summ_eval(output_folder, data_pipeline, eval_data_source, summ_gen_func,
              rev_formatter_func, sent_splitter=None, avg_rouge=True,
              analytics_func=None):
    """Performs evaluation based on Amazon and YELP summaries."""
    output_fn = "%s_eval.json" % get_file_name(eval_data_source['data_path'])
    output_path = comb_paths(output_folder, output_fn)
    logger.info("Performing the summary evaluation on %s." % eval_data_source)
    eval_proc = SummEvalProc(data_pipeline, avg_rouge=avg_rouge,
                             summs_gen_func=summ_gen_func,
                             sent_splitter=sent_splitter,
                             revs_formatter_func=rev_formatter_func,
                             analytics_func=analytics_func)
    eval_proc.eval(eval_data_source, output_file_path=output_path)


def gen_summs(data_iter, output_file_path, summ_gen_func):
    """Generates summaries and saves them to a txt file. Order is preserved"""
    out_file = open(output_file_path, encoding='utf-8', mode='w')
    for batch in data_iter:
        summs = summ_gen_func(batch)
        for summ in summs:
            out_file.write(summ + '\n')
    out_file.close()
