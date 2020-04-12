from mltoolkit.mldp.utils.constants.dp import TERMINATION_TOKEN
from mltoolkit.mldp.steps.readers import BaseReader
from mltoolkit.mldp.steps.readers.helpers import create_openers_of_valid_files, \
    populate_queue_with_chunks
from pandas.io.parsers import TextFileReader
from mltoolkit.mldp.utils.tools import DataChunk
from mltoolkit.mldp.utils.helpers.validation import validate_data_paths
from functools import partial as fun_partial
from mltoolkit.mlutils.helpers.general import listify
from multiprocessing.dummy import Pool
from queue import Queue
from copy import copy


class CsvReader(BaseReader):
    """
    CSV reader is based on a pandas internal reader (TextReader) for reading of
    raw data-chunks and inferring column types.

    Csv files must have .csv extensions, otherwise files will be ignored.

    The implementation is multi-threaded, and can assign separate files to be
    read by different threads. However, in that case data-chunks are read
    without the preserved order guarantees. Will not give any advantage if
    the pandas reading engine is not 'c'. Please refer to pandas.read_csv for
    details.

    If `chunk_size` exceeds a file's data units count, an
    incomplete data-chunk will be produced.
    """

    def __init__(self, chunk_size=1000, worker_threads_num=1, buffer_size=5,
                 name_prefix=None, engine='c', sep='\t', encoding='utf-8',
                 timeout=None, use_lists=True, **parser_kwargs):
        """
        :param chunk_size: the intermediate number of data units that are
                           passed along the pipeline. Larger data-chunks consume
                           more memory but can be beneficial for vectorized
                           operations (e.g. np.log).
        :param worker_threads_num: a number of workers that should be assigned
                                   to reading separate files. Will not be give
                                   an advantage if the pandas reading engine is
                                   not 'c'(engine='c').
                                   Please refer to pandas.read_csv for details.
        :param buffer_size: the maximum number of data-chunks that a buffer
                            queue can contain. Only used for multi-threaded
                            setup. The collector queue is populated
                            by threads. And threads freeze off-loading chunks
                            if the queue is full. They resume when the queue
                            gets free slots.
        :param engine: whether to use 'c' or 'python' modified pandas csv reader.
        :param sep: separation of columns in the input files.
        :param encoding: encoding of the input files.
        :param timeout: the number of seconds before raise the Empty Queue error
                        if multi-threading is used.
        :param use_lists: whether to use lists or numpy arrays as data-holders.
        :param parser_kwargs: additional parameters that should be passed to the
                             pandas reader (see pandas.read_csv).
        """
        super(CsvReader, self).__init__(chunk_size=chunk_size,
                                        name_prefix=name_prefix)
        if type(worker_threads_num) != int or worker_threads_num <= 0:
            raise ValueError("Please provide a valid integer for"
                             " worker_threads_number."
                             " It must be non-negative.")
        if worker_threads_num > 1 and engine != 'c':
            raise Warning("There will be no advantage in multi-threading if"
                          " the engine is not set to 'c' due to GIL.")
        if worker_threads_num == 1:
            buffer_size = None

        self.worker_threads_num = worker_threads_num
        self.parser_kwargs = parser_kwargs
        self.parser_kwargs['engine'] = engine
        self.parser_kwargs['sep'] = sep
        self.parser_kwargs['encoding'] = encoding
        self.buffer_size = buffer_size
        self.use_lists = use_lists
        self.timeout = timeout

    def _iter(self, data_path, ext='csv'):
        """
        :param data_path: a string corresponding to a location of data
                          (file or folder). If list is provided, will assume
                          multiple data paths.
        :return: generator of data-chunks.
        """
        try:
            validate_data_paths(data_path)
        except Exception as e:
            raise e

        data_paths = listify(data_path)
        file_openers = create_openers_of_valid_files(data_paths, ext=ext,
                                                     encoding=
                                                     self.parser_kwargs[
                                                         'encoding'])
        if not file_openers:
            raise ValueError(
                "No valid files to open, please check the provided"
                " 'data_path' (%s). Note that files without the '%s' extension"
                " are ignored." % (data_paths, '.csv'))

        if self.worker_threads_num > 1:
            chunk_iter = self._create_multi_th_gen(file_openers)
        else:
            chunk_iter = self._create_single_th_gen(file_openers)
        return chunk_iter

    def _create_multi_th_gen(self, file_openers):
        """
        Creates a multi-threaded generator of data-chunks by reading data from
        data_paths. Works by spawning thread workers and assigning csv files to
        them. Threads populate a queue with raw data-chunks.

        :param file_openers: a list of function that return opened files
        """
        # the queue will accum raw data-chunks produced by threads
        chunk_queue = Queue(maxsize=self.buffer_size)
        # function's partial that only will expect a file opener used by workers
        parser_kwargs = self.adjust_kwargs_to_engine(self.parser_kwargs)
        iter_creator = fun_partial(self.get_data_chunk_iter,
                                   chunksize=self.chunk_size, **parser_kwargs)
        queue_populator = fun_partial(populate_queue_with_chunks,
                                      itr_creator=iter_creator,
                                      queue=chunk_queue)

        # creating a pool of threads, and assigning jobs to them
        pool = Pool(self.worker_threads_num)
        pool.map_async(queue_populator, file_openers)
        pool.close()  # indicating that never going to submit more work

        # the inf. while loop is broken when all files are read,
        # i.e. a termination token is received for each file
        received_termin_tokens_count = 0
        while True:
            chunk = chunk_queue.get(timeout=self.timeout)
            if isinstance(chunk, Exception):
                raise chunk
            if chunk == TERMINATION_TOKEN:
                received_termin_tokens_count += 1
                if received_termin_tokens_count == len(file_openers):
                    pool.join()
                    break
            else:
                yield chunk

    def _create_single_th_gen(self, file_openers):
        """Single threaded generator that avoid using Pools and Queues."""
        parser_kwargs = self.adjust_kwargs_to_engine(self.parser_kwargs)
        for file_opener in file_openers:
            dc_iter = self.get_data_chunk_iter(file_opener,
                                               chunksize=self.chunk_size,
                                               **parser_kwargs)
            for chunk in dc_iter:
                yield chunk

    def get_data_chunk_iter(self, file_opener, **parser_kwargs):
        """
        Create and return a modified pandas data generator that spits out
        data-chunks instead of data-frames.
        """
        f = file_opener()
        try:
            pandas_iter = _TextParser(f, iterable=True,
                                      use_lists=self.use_lists, **parser_kwargs)
        except Exception as e:
            raise e
        for chunk in pandas_iter:
            yield chunk
        # TODO: new pandas doesn't close the file for some reason
        # TODO: investigate what can be done to fix that in a proper way
        f.close()

    @staticmethod
    def adjust_kwargs_to_engine(kwargs):
        """
        Makes sure that passed kwargs match the engine. As c engine requires
        a different argument for separators.
        """
        new_kwargs = copy(kwargs)  # to avoid alteration by reference
        # if new_kwargs['engine'] == 'c':
        if 'delimiter' not in new_kwargs:
            if 'sep' in kwargs:
                new_kwargs['delimiter'] = new_kwargs['sep']
                del new_kwargs['sep']
            else:
                new_kwargs['delimiter'] = ','
        return new_kwargs


class _TextParser(TextFileReader):
    """
    A modified version of the Pandas TextFileReader, which does not convert
    read data into Pandas DataFrames but instead leaves data in the dictionary
    of the numpy arrays format.
    """

    def __init__(self, f, use_lists=False, **kwargs):
        super(_TextParser, self).__init__(f, **kwargs)
        self.use_lists = use_lists

    def read(self, nrows=None):
        """
        :return: data-chunk.
        """
        if nrows is not None:
            if self.options.get('skipfooter'):
                raise ValueError('skipfooter not supported for iteration')
        fields_and_fields_to_data_tuple = self._engine.read(nrows)
        index, columns, col_dict = fields_and_fields_to_data_tuple

        dc = DataChunk()
        for cn in columns:
            dc[cn] = col_dict[cn]
            if self.use_lists:
                dc[cn] = dc[cn].tolist()
        return dc
