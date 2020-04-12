from mltoolkit.mldp.utils.constants.vocabulary import PAD, START, END, UNK
from mltoolkit.mldp import Pipeline
from mltoolkit.mldp.steps.readers import CsvReader
from mltoolkit.mldp.steps.transformers.nlp import TokenProcessor, VocabMapper, \
    SeqLenComputer, Padder, SeqWrapper
from mltoolkit.mldp.steps.transformers.general import ChunkSorter
from mltoolkit.mldp.steps.transformers.field import FieldRenamer
from copycat.data_pipelines.steps import YelpTransformer, RevMapper, \
    AmazonTransformer
from mltoolkit.mldp.steps.formatters import PyTorchFormatter
from copycat.utils.fields import ModelF, AmazonEvalF, YelpEvalF

from copycat.data_pipelines.steps import GoldSummRevIndxsCreator


def assemble_eval_pipeline(word_vocab, max_groups_per_chunk=1, dataset='yelp',
                           tokenization_func=lambda x: x.split()):
    """Assembles the pipeline for evaluation on the YELP and Amazon eval set."""
    assert dataset in ['yelp', 'amazon']

    if dataset == 'yelp':
        fields_obj = YelpEvalF
        fname_renamer = FieldRenamer({fields_obj.BUSINESS_ID: ModelF.GROUP_ID})
        dataset_spec_trans = YelpTransformer()
    else:
        fields_obj = AmazonEvalF
        fname_renamer = FieldRenamer({fields_obj.PROD_ID: ModelF.GROUP_ID,
                                      fields_obj.CAT: ModelF.CAT})
        dataset_spec_trans = AmazonTransformer()

    assert START in word_vocab
    assert END in word_vocab

    reader = CsvReader(sep='\t', encoding='utf-8',
                       engine='python', quotechar='\'',
                       chunk_size=max_groups_per_chunk)

    # notice that I do not tokenize summaries, I leave them as they are!
    token_processor = TokenProcessor(fnames=fields_obj.REVS,
                                     tokenization_func=tokenization_func)
    # notice that I don't convert summs tokens to ids
    vocab_mapper = VocabMapper({fn: word_vocab for fn in fields_obj.REVS})

    seq_wrapper = SeqWrapper(ModelF.REV, start_el=word_vocab[START].id,
                             end_el=word_vocab[END].id)

    seq_len_computer = SeqLenComputer(ModelF.REV, ModelF.REV_LEN)

    padder = Padder(fname=ModelF.REV, new_mask_fname=ModelF.REV_MASK,
                    pad_symbol=word_vocab[PAD].id, padding_mode='right')

    sorter = ChunkSorter(field_name=ModelF.REV_LEN,
                         fields_to_sort=[ModelF.REV, ModelF.REV_MASK,
                                         ModelF.CAT, ModelF.GROUP_ID])

    indxs_creator = GoldSummRevIndxsCreator(group_id_fname=ModelF.GROUP_ID)

    rev_mapper = RevMapper(group_rev_indxs_fname=ModelF.GROUP_REV_INDXS,
                           group_rev_mask_fname=ModelF.GROUP_REV_INDXS_MASK,
                           rev_mask_fname=ModelF.REV_MASK)

    formatter = PyTorchFormatter()

    pipeline = Pipeline(reader=reader, error_on_invalid_chunk=False)

    pipeline.add_step(token_processor)
    pipeline.add_step(vocab_mapper)

    pipeline.add_step(dataset_spec_trans)
    pipeline.add_step(fname_renamer)

    pipeline.add_step(seq_wrapper)
    pipeline.add_step(seq_len_computer)
    pipeline.add_step(padder)
    pipeline.add_step(sorter)
    pipeline.add_step(indxs_creator)
    pipeline.add_step(rev_mapper)

    pipeline.add_step(formatter)

    return pipeline
