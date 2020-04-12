from mltoolkit.mldp import PyTorchPipeline
from mltoolkit.mldp.steps.general import ChunkAccumulator
from mltoolkit.mldp.steps.collectors import UnitSampler, ChunkCollector, \
    ChunkShuffler
from mltoolkit.mldp.steps.readers import CsvReader
from mltoolkit.mldp.steps.transformers.general import Postfixer, ChunkSorter
from mltoolkit.mldp.steps.transformers.nlp import TokenProcessor, VocabMapper, \
    SeqLenComputer, SeqWrapper, Padder
from mltoolkit.mldp.steps.transformers.field import FieldRenamer
from mltoolkit.mldp.utils.constants.vocabulary import START, END, PAD
from copycat.data_pipelines.steps import GroupFileShuffler, SummRevIndxsCreator, \
    RevMapper
from copycat.utils.fields import InpDataF, ModelF
from csv import QUOTE_NONE


def assemble_train_pipeline(word_vocab, max_groups_per_batch=1,
                            min_revs_per_group=None, max_revs_per_group=10,
                            seed=None, workers=1):
    """
    This pipeline is specific to the preprocessed Amazon and Yelp reviews.
    Creates a flow of transformation steps that modify the data until the final
    form is reached in terms of PyTorch tensors.

    :param word_vocab: vocabulary object with words/tokens.
    :param max_groups_per_batch: number of groups each batch should have.
    :param min_revs_per_group: number of reviews a group should have in order
                               not to be discarded.
    :param max_revs_per_group: self-explanatory.
    :param reseed: set it to True if use multi-processing and want it to return
                   different sequences of batches every epoch. This has to do
                   purely with multi-processing issues in combination with
                   numpy.
    """
    assert START in word_vocab
    assert END in word_vocab

    group_files_shuffler = GroupFileShuffler()

    reader = CsvReader(sep='\t', engine='python', chunk_size=None,
                       encoding='utf-8', quoting=QUOTE_NONE,
                       timeout=None, worker_threads_num=1)

    fname_renamer = FieldRenamer({InpDataF.REV_TEXT: ModelF.REV,
                                  InpDataF.GROUP_ID: ModelF.GROUP_ID})

    unit_sampler = UnitSampler(id_fname=ModelF.GROUP_ID,
                               sample_all=True,
                               min_units=min_revs_per_group,
                               max_units=max_revs_per_group)
    unit_sampler_accum = ChunkAccumulator(unit_sampler)

    # since we're splitting one group into multiple chunks, it's convenient
    # to postfix each group_id name, such that it would be possible to
    # associate summaries with different subsets of reviews
    postfixer = Postfixer(id_fname=ModelF.GROUP_ID)

    # to avoid having same product/business appearing in the same merged
    # data-chunk, buffer a small number of them, shuffle, and release
    chunk_shuffler = ChunkAccumulator(ChunkShuffler(buffer_size=500))

    # accumulates a fixed number of group chunks, merges them
    # together, and passes along the pipeline
    chunk_coll = ChunkCollector(buffer_size=max_groups_per_batch)
    chunk_accum = ChunkAccumulator(chunk_coll)

    # alternation of data entries
    tokenizer = TokenProcessor(fnames=ModelF.REV)
    vocab_mapper = VocabMapper({ModelF.REV: word_vocab})

    seq_wrapper = SeqWrapper(fname=ModelF.REV,
                             start_el=word_vocab[START].id,
                             end_el=word_vocab[END].id)

    seq_len_computer = SeqLenComputer(ModelF.REV, ModelF.REV_LEN)

    sorter = ChunkSorter(ModelF.REV_LEN)

    padder = Padder(fname=ModelF.REV,
                    new_mask_fname=ModelF.REV_MASK,
                    pad_symbol=word_vocab[PAD].id, padding_mode='right')

    summ_rev_indxs_creator = SummRevIndxsCreator(group_id_fname=ModelF.GROUP_ID,
                                                 category_fname=ModelF.CAT)

    rev_mapper = RevMapper(group_rev_indxs_fname=ModelF.GROUP_REV_INDXS,
                           group_rev_mask_fname=ModelF.GROUP_REV_INDXS_MASK,
                           rev_mask_fname=ModelF.REV_MASK)

    pipeline = PyTorchPipeline(reader=reader, preprocessor=group_files_shuffler,
                               worker_processes_num=workers, seed=seed,
                               error_on_invalid_chunk=False, timeout=None)

    pipeline.add_step(fname_renamer)

    pipeline.add_step(unit_sampler_accum)
    pipeline.add_step(postfixer)
    pipeline.add_step(chunk_shuffler)
    pipeline.add_step(chunk_accum)

    # entry transformations
    pipeline.add_step(tokenizer)
    pipeline.add_step(vocab_mapper)
    pipeline.add_step(seq_wrapper)
    pipeline.add_step(seq_len_computer)
    pipeline.add_step(sorter)
    pipeline.add_step(padder)

    # adding additional fields for attention and summarization
    pipeline.add_step(summ_rev_indxs_creator)
    pipeline.add_step(rev_mapper)

    return pipeline
