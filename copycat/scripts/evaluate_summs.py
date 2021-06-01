from argparse import ArgumentParser
import pandas as pd
from copycat.eval.metrics.google_rouge.google_rouge import GoogleRouge
from copycat.utils.helpers.io import read_text_file
from mltoolkit.mlutils.helpers.formatting.general import metrics_to_str


def evaluate_summs(gen_summ_file_path, gold_summ_file_path, gold_summ_fnames):
    """
    Evaluates generated versus true summaries using Google ROUGE script.

    :param gen_summ_file_path: txt file path to generated summaries. Each
        summary should be on a separate line.
    :param gold_summ_file_path: csv file path to gold summaries.
    :param gold_summ_fnames: list of CSV columns to gold summaries.
    """

    gen_summs = read_text_file(gen_summ_file_path)
    rouge_scorer = GoogleRouge()

    gold_ds = pd.read_csv(gold_summ_file_path, sep="\t", quotechar="\'",
                          encoding='utf-8')
    assert len(gen_summs) == len(gold_ds)

    gold_summs = [[du[gsumm_fname] for gsumm_fname in gold_summ_fnames]
    for _, du in gold_ds.iterrows()]

    for _gen_summ, _gold_summs in zip(gen_summs, gold_summs):
        rouge_scorer.accum([_gen_summ], [_gold_summs])
    res = rouge_scorer.aggr()

    for k, v in res.items():
        print("%s %s" % (k, metrics_to_str(v)))


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--gen-summ-file-path', required=True)
    parser.add_argument('--gold-summ-file-path', required=True)
    parser.add_argument('--gold-summ-fnames', nargs="+", required=True)
    evaluate_summs(**vars(parser.parse_args()))
