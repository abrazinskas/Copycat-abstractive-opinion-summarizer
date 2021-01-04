from mltoolkit.mlmo.utils.tools import BaseHP
from mltoolkit.mlutils.helpers.paths_and_files import comb_paths
from mltoolkit.mlutils.tools import ExperimentsPathController
from sacremoses import MosesTokenizer, MosesDetokenizer, MosesTruecaser
from functools import partial
import nltk
from mltoolkit.mlmo.utils.helpers.analytics import ngram_seq_analysis


class RunHP(BaseHP):
    """Contains configuration parameters for running the model."""

    def __init__(self):
        super(RunHP, self).__init__()

        #   GENERAL  #
        self.seed = 42
        self.cuda_device_id = 0
        self.device = 'cuda'  # 'cuda' or 'cpu'
        self.training_logging_step = 50  # that often to print internal metrics
        self.epochs = 10  # if set to 0 will only perform evaluation
        self.learning_rate = 0.0005
        self.grads_clip = 0.25

        # GENERAL DATA RELATED #
        self.dataset = 'amazon'
        self.train_max_groups_per_batch = 6
        self.val_max_groups_per_batch = 13
        self.eval_max_groups_per_batch = 20
        self.max_rev_per_group = 8

        #   DATA SOURCES  #
        # `early_term` limits the number of chunks per epoch
        self.train_early_term = None
        self.val_early_term = None
        self.gener_early_term = 2

        #  GENERAL PATHS   #
        self.root_path = 'copycat'
        self.experiments_folder = 'first_run'
        self.output_dir = f'{self.root_path}/runs/{self.dataset}/{self.experiments_folder}'
        self.checkpoint_full_fn = 'checkpoint.tar'
        epc = ExperimentsPathController()
        self.output_path = epc(self.output_dir)
        self.checkpoint_path = f'{self.root_path}/artifacts/{self.dataset}/checkpoint.tar'
        self.tcaser_model_path = f'{self.root_path}/artifacts/{self.dataset}/data/tcaser.model'

        #   DATA PATHS  #
        self.base_data_path = f'data/{self.dataset}/'
        self.train_fp = comb_paths(self.base_data_path, "split/train/")
        self.val_fp = comb_paths(self.base_data_path, 'split/val/')
        self.words_vocab_fp = f'{self.root_path}/artifacts/{self.dataset}/data/words.txt'
        self.eval_dev_fp = comb_paths(self.base_data_path, 'gold', 'val.csv')
        self.eval_test_fp = comb_paths(self.base_data_path, 'gold', 'test.csv')

        #   ANNEALING   #
        self.c_m = 8.
        self.c_r = 0.8
        self.c_kl_ann_max_val = 1.
        self.c_kl_ann_batches = self.epochs * self.train_early_term if self.train_early_term else self.epochs * 10000
        self.z_m = 8.
        self.z_c = 0.8
        self.z_kl_ann_max_val = 1.
        self.z_kl_ann_batches = self.epochs * self.train_early_term if self.train_early_term else self.epochs * 10000

        #   DECODING/GENERATION  #
        self.beam_size = 5
        self.beam_len_norm = True
        self.beam_excl_words = []
        self.block_ngram_repeat = 3  # or None
        self.ngram_mirror_window = 3  # or None
        self.mirror_conjs = ["and", 'or', ',', 'but']  # or None
        self.block_consecutive = True
        self.min_gen_seq_len = 20

        #   POST-PROCESSING AND ANALYTICS #
        mt = MosesTokenizer()
        self.tok_func = partial(mt.tokenize, escape=False)
        self.sent_split_func = nltk.sent_tokenize
        dt = MosesDetokenizer()
        self.detok_func = partial(dt.detokenize, unescape=False)
        true_caser = MosesTruecaser(load_from=self.tcaser_model_path,
                                    is_asr=True)
        self.true_case_func = partial(true_caser.truecase, return_str=True,
                                      use_known=True)
        self.analytics_func = partial(ngram_seq_analysis,
                                      tokenizer=self.tok_func,
                                      sent_splitter=self.sent_split_func,
                                      n_grams_to_comp=(2, 3, 4))
