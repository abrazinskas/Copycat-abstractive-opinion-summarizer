# this file contains a workflow that creates necessary objects and executes
# methods for training and evaluation.
from copycat.data_pipelines.assemblers import assemble_train_pipeline, \
    assemble_eval_pipeline, assemble_vocab_pipeline, assemble_infer_pipeline
from mltoolkit.mldp.utils.tools import Vocabulary
from mltoolkit.mldp.utils.constants.vocabulary import START, END, PAD
from mltoolkit.mlutils.helpers.logging import init_logger, DEBUG, INFO
from mltoolkit.mlutils.helpers.paths_and_files import comb_paths, get_file_name
from copycat.modelling.interfaces import IDevCopyCat as IDev, ICopyCat as IModel
from copycat.modelling import CopyCat as Model
from mltoolkit.mlmo.generation import Beamer
from copycat.utils.fields import InpDataF
from torch.nn.init import xavier_uniform_, normal_
from mltoolkit.mlutils.helpers.formatting.general import format_big_box
from copycat.utils.helpers.run import gen_seqs, summ_eval, gen_summs
from copycat.utils.helpers.io import get_rev_number
from time import time
from torch import manual_seed
from functools import partial
import numpy as np
import os
from mltoolkit.mlmo.utils.tools.annealing import KlCycAnnealing
from copycat.utils.hparams import ModelHP, RunHP
from copycat.utils.tools import SeqPostProcessor
from copycat.utils.constants import VOCAB_DEFAULT_SYMBOLS
from argparse import ArgumentParser


#   PARSER   #

parser = ArgumentParser()
parser.add_argument('--infer-input-file-path', type=str,
                    help="CSV file with data for inference.")
parser.add_argument('--infer-batch-size', type=int, default=10,
                    help='The batch size used in inference.')

parser_args = parser.parse_args()

model_hp = ModelHP()
run_hp = RunHP()
manual_seed(run_hp.seed)
np.random.seed(run_hp.seed)

#   DATA SOURCES    #

vocab_data_source = {"data_path": run_hp.train_fp}
train_data_source = {"data_path": run_hp.train_fp,
                     "early_term": run_hp.train_early_term}
val_data_source = {"data_path": run_hp.val_fp,
                   'early_term': run_hp.val_early_term}
eval_dev_data_source = {"data_path": run_hp.eval_dev_fp}
eval_test_data_source = {"data_path": run_hp.eval_test_fp}

gen_data_sources = [
    {"data_path": run_hp.train_fp, 'early_term': run_hp.gener_early_term},
    {"data_path": run_hp.val_fp, 'early_term': run_hp.gener_early_term}
]

os.environ['CUDA_VISIBLE_DEVICES'] = str(run_hp.cuda_device_id)

logger = init_logger(logger_name="", level=INFO,
                     output_path=comb_paths(run_hp.output_path, "log.txt"))
logger.info('CUDA_VISIBLE_DEVICES=%s' % os.environ.get('CUDA_VISIBLE_DEVICES'))

#   KL ANNEALING MECHANISMS  #

c_kl_ann = KlCycAnnealing(t=run_hp.c_kl_ann_batches, m=run_hp.c_m,
                          r=run_hp.c_r, max_val=run_hp.c_kl_ann_max_val)
z_kl_ann = KlCycAnnealing(t=run_hp.z_kl_ann_batches, m=run_hp.z_m,
                          r=run_hp.c_r, max_val=run_hp.z_kl_ann_max_val)

#   PIPELINES AND VOCAB   #

vocab_pipeline = assemble_vocab_pipeline(text_fname=InpDataF.REV_TEXT)
word_vocab = Vocabulary(vocab_pipeline, name_prefix="word")

# adding special symbols before creating vocab, so they would appear on top
for st in VOCAB_DEFAULT_SYMBOLS:
    if st not in word_vocab:
        word_vocab.add_special_symbol(st)

word_vocab.load_or_create(run_hp.words_vocab_fp,
                          data_source=vocab_data_source,
                          max_size=model_hp.ext_vocab_size, sep=' ',
                          data_fnames=InpDataF.REV_TEXT)

word_vocab.write(comb_paths(run_hp.output_path, "word_vocab.txt"), sep=' ')

train_pipeline = assemble_train_pipeline(word_vocab,
                                         max_groups_per_batch=run_hp.train_max_groups_per_batch,
                                         min_revs_per_group=run_hp.max_rev_per_group,
                                         max_revs_per_group=run_hp.max_rev_per_group,
                                         seed=None, workers=1)
val_pipeline = assemble_train_pipeline(word_vocab,
                                       max_groups_per_batch=run_hp.val_max_groups_per_batch,
                                       min_revs_per_group=run_hp.max_rev_per_group,
                                       max_revs_per_group=run_hp.max_rev_per_group,
                                       seed=run_hp.seed, workers=1)

eval_data_pipeline = assemble_eval_pipeline(word_vocab,
                                            dataset=run_hp.dataset,
                                            tokenization_func=run_hp.tok_func,
                                            max_groups_per_chunk=run_hp.eval_max_groups_per_batch)


#   MODEL AND INTERFACES INITIALIZATION   #

summ_post_proc = SeqPostProcessor(tokenizer=lambda x: x.split(),
                                  detokenizer=run_hp.detok_func,
                                  sent_splitter=run_hp.sent_split_func,
                                  tcaser=run_hp.true_case_func)

start_id = word_vocab[START].id
end_id = word_vocab[END].id
pad_id = word_vocab[PAD].id

model = Model(**model_hp.to_dict())

beamer = Beamer(model.decode_beam,
                start_id=start_id, end_id=end_id, n_best=1,
                device=run_hp.device, len_norm=run_hp.beam_len_norm,
                excl_ids=[word_vocab[w].id for w in run_hp.beam_excl_words],
                block_ngram_repeat=run_hp.block_ngram_repeat,
                ngram_mirror_window=run_hp.ngram_mirror_window,
                mirror_conj_ids=[word_vocab[conj].id for conj in
                                 run_hp.mirror_conjs] if run_hp.mirror_conjs is not None else None,
                beam_size=run_hp.beam_size,
                block_consecutive=run_hp.block_consecutive)

imodel = IModel(model=model, grads_clip=run_hp.grads_clip,
                device=run_hp.device, min_gen_seq_len=run_hp.min_gen_seq_len,
                learning_rate=run_hp.learning_rate, beamer=beamer)

idev = IDev(imodel=imodel, train_data_pipeline=train_pipeline,
            val_data_pipeline=val_pipeline, word_vocab=word_vocab,
            c_kl_ann=c_kl_ann, z_kl_ann=z_kl_ann,
            tok_func=run_hp.tok_func,
            detok_func=run_hp.detok_func,
            sent_split_func=run_hp.sent_split_func)

#   PARAMETERS LOADING OR INITIALIZATION    #

if run_hp.checkpoint_path:
    imodel.load_state(run_hp.checkpoint_path, strict=True)
else:
    imodel.init_weights(multi_dim_init_func=xavier_uniform_,
                        single_dim_init_func=lambda x: normal_(x, std=0.1))

idev.save_setup_str(run_hp.output_path, run_hp.experiments_descr)

# logging and saving hyper-params
logger.info(format_big_box(str(run_hp)))
logger.info(format_big_box(str(model_hp)))
logger.info("Trainable parameters: %d." % sum(
    p.numel() for p in model.parameters() if p.requires_grad))
run_hp.save(comb_paths(run_hp.output_path, 'run_hp.json'))
model_hp.save(comb_paths(run_hp.output_path, 'model_hp.json'))

#   TRAINING PROCEDURE  #

if parser_args.infer_input_file_path is None:
    if run_hp.epochs > 0:
        gen_func = partial(idev.gen_and_save_summs, beam_size=run_hp.beam_size)

        def after_ep_func(epoch):
            out_fp = comb_paths(run_hp.output_path,
                                'ep%d_%s' % (epoch, run_hp.checkpoint_full_fn))
            imodel.save_state(out_fp)

            gen_folder_path = comb_paths(run_hp.output_path,
                                         "output_ep%d" % epoch)
            summ_eval(output_folder=gen_folder_path,
                      data_pipeline=eval_data_pipeline,
                      eval_data_source=eval_dev_data_source,
                      summ_gen_func=partial(idev.summ_generator,
                                            summ_post_proc=summ_post_proc),
                      rev_formatter_func=idev.format_revs,
                      avg_rouge=True,
                      sent_splitter=run_hp.sent_split_func,
                      analytics_func=run_hp.analytics_func)
            gen_seqs(data_sources=gen_data_sources,
                     output_folder=gen_folder_path,
                     gen_func=partial(idev.gen_and_save_summs))


        start = time()
        idev.standard_workflow(train_data_source=train_data_source,
                               val_data_source=val_data_source,
                               logging_period=run_hp.training_logging_step,
                               epochs=run_hp.epochs,
                               after_epoch_func=after_ep_func)
        logger.info("Total time elapsed %f (s) " % (time() - start))

        imodel.save_state(comb_paths(run_hp.output_path,
                                     run_hp.checkpoint_full_fn))

    #   AFTER TRAINING PROCEDURES   #

    gen_folder_path = comb_paths(run_hp.output_path, "output")

    summ_eval(output_folder=gen_folder_path, eval_data_source=eval_test_data_source,
              summ_gen_func=partial(idev.summ_generator,
                                    summ_post_proc=summ_post_proc),
              data_pipeline=eval_data_pipeline,
              rev_formatter_func=idev.format_revs,
              avg_rouge=True,
              sent_splitter=run_hp.sent_split_func,
              analytics_func=run_hp.analytics_func)

    gen_seqs(data_sources=gen_data_sources, output_folder=gen_folder_path,
             gen_func=partial(idev.gen_and_save_summs))
else:
    # inference procedure where summaries are generated for reviews in CSV
    # files
    infer_bsz = parser_args.infer_batch_size
    infer_inp_file_path = parser_args.infer_input_file_path
    out_file_name = get_file_name(infer_inp_file_path)
    infer_out_file_path = comb_paths(run_hp.output_path,
                                     f'{out_file_name}.out.txt')

    assert infer_inp_file_path is not None
    rev_num = get_rev_number(infer_inp_file_path)

    logger.info("Performing inference/summary generation")
    infer_data_pipeline = assemble_infer_pipeline(word_vocab, max_reviews=rev_num,
                                                  tokenization_func=run_hp.tok_func,
                                                  max_groups_per_chunk=infer_bsz)
    summ_pproc = SeqPostProcessor(tokenizer=lambda x: x.split(),
                                  detokenizer=run_hp.detok_func,
                                  tcaser=run_hp.true_case_func)

    logger.info(f"Saving summaries to: '{infer_out_file_path}'")
    gen_summs(infer_data_pipeline.iter(data_path=infer_inp_file_path),
              output_file_path=infer_out_file_path,
              summ_gen_func=partial(idev.summ_generator,
                                    summ_post_proc=summ_post_proc))
