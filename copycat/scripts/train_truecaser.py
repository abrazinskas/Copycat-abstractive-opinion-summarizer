from argparse import ArgumentParser
from sacremoses import MosesTruecaser
from mltoolkit.mldp.steps.readers import CsvReader
from mltoolkit.mlutils.helpers.paths_and_files import safe_mkfdir
from mltoolkit.mlutils.helpers.logging import init_logger
from copycat.utils.fields import InpDataF
from csv import QUOTE_NONE
import os

logger = init_logger("")


def train_and_save_true_casing_model(data_path, tcaser_fp,
                                     text_fname=InpDataF.REV_TEXT):
    """Trains the Moses model on tokenized csv files; saves params."""
    mtr = MosesTruecaser(is_asr=True)
    reader = CsvReader(quoting=QUOTE_NONE, sep='\t', engine='python',
                       encoding='utf-8')
    texts = []
    logger.info("Loading data from '%s'." % data_path)
    for dc in reader.iter(data_path=data_path):
        for du in dc.iter():
            texts.append(du[text_fname].split())
    logger.info("Loaded the data.")
    safe_mkfdir(tcaser_fp)
    logger.info("Training the truecaser.")
    mtr.train(texts, save_to=tcaser_fp, progress_bar=True, processes=1)
    logger.info("Done, saved the truecaser to `%s`." % tcaser_fp)


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument("--data_path", nargs="+")
    parser.add_argument("--text_fname", default=InpDataF.REV_TEXT)
    parser.add_argument("--tcaser_fp",
                        help='File path where the truecaser should be saved.')
    train_and_save_true_casing_model(**vars(parser.parse_args()))
