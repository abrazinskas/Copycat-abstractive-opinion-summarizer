from mltoolkit.mldp.utils.constants.vocabulary import PAD, START, END, UNK
from mltoolkit.mldp import PyTorchPipeline
from mltoolkit.mldp.steps.readers import CsvReader
from mltoolkit.mldp.steps.transformers.nlp import TokenProcessor, VocabMapper, \
    SeqLenComputer, Padder, SeqWrapper
from mltoolkit.mldp.steps.transformers.general import ChunkSorter
from copycat.data_pipelines.steps import ReviewFlattener,\
    GoldSummRevIndxsCreator
from copycat.utils.fields import ModelF, InfDataF


def assemble_infer_pipeline(word_vocab, max_groups_per_chunk=1, max_reviews=10,
                            tokenization_func=lambda x: x.split()):
    """Assembles a simple inference pipeline for summary generation. Assumes
    that csv files are read where reviews have the following column names:
    'rev1', 'rev2', ..., 'revN', each review separated by \t.

    Args:
        word_vocab: word vocabulary to convert words to ids.
        max_groups_per_chunk: self-explanatory.
        max_reviews: the maximum number of reviews to load per group. Columns
            in the CSV file should be `rev1`, ...., `revN`.
        tokenization_func: self-explanatory.
    """
    rev_fnames = [f'{InfDataF.REV_PREFIX}{i}' for i in range(1, max_reviews + 1)]

    assert START in word_vocab
    assert END in word_vocab

    reader = CsvReader(sep='\t', encoding='utf-8', engine='python',
                       quotechar='\'', chunk_size=max_groups_per_chunk)

    rev_flattener = ReviewFlattener(group_id_fname=InfDataF.GROUP_ID,
                                    rev_fnames=rev_fnames)

    token_processor = TokenProcessor(fnames=ModelF.REV,
                                     tokenization_func=tokenization_func)
    # notice that I don't convert summs tokens to ids
    vocab_mapper = VocabMapper({ModelF.REV: word_vocab})

    seq_wrapper = SeqWrapper(ModelF.REV, start_el=word_vocab[START].id,
                             end_el=word_vocab[END].id)

    seq_len_computer = SeqLenComputer(ModelF.REV, ModelF.REV_LEN)

    padder = Padder(fname=ModelF.REV, new_mask_fname=ModelF.REV_MASK,
                    pad_symbol=word_vocab[PAD].id, padding_mode='right')

    sorter = ChunkSorter(field_name=ModelF.REV_LEN,
                         fields_to_sort=[ModelF.REV, ModelF.GROUP_ID])

    # re-using the step
    summ_rev_indx_creator = GoldSummRevIndxsCreator(group_id_fname=ModelF.GROUP_ID)

    pipeline = PyTorchPipeline(reader=reader, error_on_invalid_chunk=False)

    pipeline.add_step(rev_flattener)
    pipeline.add_step(token_processor)
    pipeline.add_step(vocab_mapper)

    pipeline.add_step(seq_wrapper)
    pipeline.add_step(seq_len_computer)
    pipeline.add_step(padder)
    pipeline.add_step(sorter)

    pipeline.add_step(summ_rev_indx_creator)

    return pipeline
