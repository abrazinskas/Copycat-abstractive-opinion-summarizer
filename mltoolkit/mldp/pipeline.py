from multiprocessing import Process, Queue, Event
from mltoolkit.mldp.steps.readers.base_reader import BaseReader
from mltoolkit.mldp.steps.base_step import BaseStep
from mltoolkit.mldp.steps.general import BaseGeneral
from mltoolkit.mldp.steps.preprocessors import BasePreProcessor
from mltoolkit.mldp.steps.formatters.base_formatter import BaseFormatter
from mltoolkit.mldp.steps.general.chunk_accumulator import ChunkAccumulator
from mltoolkit.mldp.utils.constants.dp import TERMINATION_TOKEN, EMPTY_CHUNK
from mltoolkit.mldp.utils.errors import DataChunkError
from mltoolkit.mldp.utils.helpers.validation import equal_to_constant
from mltoolkit.mldp.utils.tools import DataChunk
from mltoolkit.mlutils.helpers.multi_processing import create_gen_from_queue, \
    get_os_seed
from mltoolkit.mlutils.tools import SignedObject
from mltoolkit.mlutils.helpers.formatting.general import \
    format_to_standard_msg_str, \
    format_title
import warnings
import logging
import numpy as np
import os

logger_name = os.path.basename(__file__)
logger = logging.getLogger(logger_name)


class Pipeline(SignedObject):
    """
    The class is responsible for collecting data-chunks from a reader and
    passing them along the processing pipeline and applying processing
    components/steps in a pre-specified sequential order.

    The pipeline is designed to be used for computationally intensive real-time
    processing of data from arbitrary locations and formats. One also can easily
    take advantage of the parallel CPU architecture.

    Example pipeline:
                  [0. reader] ->
                  [1. log scaling of features (transformer)] ->
                  [2. mapping of categorical features to ids (transformer)] ->
                  [3. to pandas data-frames (formatter)]

    The pipeline will start by reading raw data (e.g. from local csv files) via
    the reader(0.), apply the chain of steps(1., 2.) to data-chunks,
    and finally produce pandas data-frames via the formatter (3.).

    The pipeline has an intermediate format of data-chunks that are passed along
    , which is essentially a dictionary of numpy arrays. The restriction allows
     to re-use pre-implemented steps. However, the user can use a formatter to
    convert data-chunks to a desired format in the end of the pipeline
    (e.g. to pandas data frames). The formatter must be the last step, as it
    outputs data-chunks of arbitrary format. The validation is performed in this 
    class, and can be turned off by passing raise_on_invalid_chunk='none' param.

    The pipeline has three architectures which are created depending on
    worker_processes_number. Each are more suitable for different cases.

    1. worker_processes_number == 0:
        the whole logic is executed on the main processes (both the reader and
        processing steps). The data-chunks are read and processed on demand.

        Most suited for predictions on very small batches or real-time due to
        low overhead.

    2. worker_processes_number == 1:
       A separate process is spawned, and the reader and processing steps
       are assigned to it. Will populate a buffer queue with processed
       data_chunks in a 'lazy' way.

       Well suited for cases when execution of processing steps on data-chunks
       is not very time consuming. However, one does not want to wait for
       processed data-chunks to be created on request.

    3. worker_processes_number > 1:
        A separate process is spawn for the reader, and the rest are spawned
        for data-chunk processing. Namely, each remaining process will have an
        independent copy of the processing steps pipeline. The reader process
        will store raw data-chunks into an input buffer queue from which the
        remaining processes will feed. The processed data-chunks will be stored
        to the output buffer queue.

        Well suited for cases when the processing steps execution on raw
        data-chunks is expensive (e.g. steps contain for-loops), and out-weight
        significantly the reading time of raw data-chunks. In addition,
        it's more beneficial if the hardware is highly parallel (multi-CPU,
        multi-core).

        The downside is that two processing queues are used, and that introduces
        an extra overhead due to serialization and de-serialization, which in
        some cases can out-weight the multi-processing benefits.
    """

    def __init__(self, reader, preprocessor=None, worker_processes_num=0,
                 input_buffer_size=5, output_buffer_size=5, seed=None,
                 error_on_invalid_chunk=True, timeout=5, name_prefix=""):
        """
        :param reader: the reader object(subclass of BaseReader) that serves
                       raw data-chunks from a source.
        :param preprocessor: an object(subclass of BasePreprocessor) that
                             contains logic to be executed before data processing
                             starts. For example, it might download data to the
                             local storage, or shuffle data.
        :param worker_processes_num: the number of processes that should be
                                        spawned to read and process data-chunk.
                                        Will affect the architecture of the
                                        pipeline. See the class docstring for
                                        more info.
        :param input_buffer_size: the maximum number of raw data-chunks that are
                                  buffered from the reader
                                  (which runs on a separate process).
                                  Only used when worker_processes_number > 1
        :param output_buffer_size: the maximum number of processed data-chunks
                                   to be accumulated before freezing processing
                                   workers. Only used when
                                   worker_processes_number >= 1.
        :param seed: if provided, it will be set every time the `iter` method
                     is called. Otherwise, the OS based seed will be set to
                     ensure randomization on separate processes. Note
                     that when multiple processes are used, it becomes impossible
                     to guarantee the same order of chunks storage to the queue.
        :param error_on_invalid_chunk: whether to raise an error if a
                                       data-chunk is detected to be invalid.
        :param timeout: after how many seconds before multiprocessing queue
                        should return an error if it's empty.
        :param name_prefix: will be added to msg title in  __str__ out if
                            provided.
        """
        if worker_processes_num < 0:
            raise ValueError("worker_processes_num must be a "
                             "non-negative integer.")

        mess_template = "The provided %s is not valid, as it's not the %s's" \
                        " subclass."
        if not isinstance(reader, BaseReader):
            raise ValueError(mess_template % "reader", BaseReader.__name__)
        if preprocessor is not None and not isinstance(preprocessor,
                                                       BasePreProcessor):
            raise ValueError(mess_template % "preprocessor",
                             BasePreProcessor.__name__)

        super(Pipeline, self).__init__(name_prefix=name_prefix)
        self.worker_processes_number = worker_processes_num
        self.preprocessor = preprocessor
        self.input_buffer_size = input_buffer_size
        self.output_buffer_size = output_buffer_size
        self.seed = seed
        self.error_on_invalid_chunk = error_on_invalid_chunk
        self.reader = reader
        self.timeout = timeout
        self.steps = []

        self._workers = []
        self._adjust_scraper()

    def add_step(self, step):
        """
        Adds a new processing component to the end of the chain of processing
        steps.

        :param step: transformer, formatter, or chunk_accumulator.
        """
        if not isinstance(step, BaseStep):
            raise ValueError(
                "The passed step is invalid. It must either be a"
                " transformer (subclass of BaseTransformer),"
                " formatter (subclass of BaseFormatter), or a "
                " chunk_accumulator."
            )
        # check if the previous step is a formatter, and prevent adding any
        # other steps, as a formatter must be the last step in the chain of
        # steps
        if len(self.steps) > 0 and isinstance(self.steps[-1], BaseFormatter):
            raise ValueError("Can't add a new step because the last one is"
                             " a formatter step, which must be the last one in"
                             " the sequence.")
        self.steps.append(step)

    def iter(self, early_term=None, **kwargs):
        """
        Creates a generator of processed data-chunks that a user can access.
        The way data-chunks are generated/processed depends on the
        workers_processes_number, see the class's docstring.

        :param early_term: if int provided, will yield a fixed number of
                             chunks before termination.
        :param kwargs: params to be passed to pre_processor(opt) or reader.
        :return: generator of processed data-chunks.
        """
        seed = self.seed if self.seed is not None else get_os_seed()
        np.random.seed(seed)

        if self.preprocessor is not None:
            kwargs = self.preprocessor(**kwargs)

        reader_gen = self.reader.iter(**kwargs)
        dc_gen = self._create_processed_data_chunks_gen(reader_gen)

        for i, dc in enumerate(dc_gen, 1):
            if early_term is not None and i > early_term:
                self._terminate_workers()
                break
            yield dc

    def _create_processed_data_chunks_gen(self, reader_gen):
        """
        :return: generator of processed data-chunks.
        """
        if self.worker_processes_number == 0:
            itr = self._create_single_process_gen(reader_gen)
        else:
            itr = self._create_multi_process_gen(reader_gen)
        return itr

    def _create_single_process_gen(self, data_producer):
        """
        Chains reader and steps together, such that the data processing would be
        performed on the main process.

        :return: generator of processed data-chunks.
        """
        return combine_steps_into_chain(data_producer=data_producer,
                                        processing_steps=self.steps,
                                        error_on_invalid_chunk=self.error_on_invalid_chunk)

    def _create_multi_process_gen(self, reader_gen):
        """
        Chains reader and steps together, such that the data processing would
        performed on multiple processes.

        It will create different architectures for data reading and processing
        depending on worker_processes_number. See the class doc string for
        more info.
        """
        term_tokens_received = 0
        output_queue = Queue(self.output_buffer_size)
        self._workers = []

        if self.worker_processes_number > 1:
            term_tokens_expected = self.worker_processes_number - 1
            input_queue = Queue(self.input_buffer_size)
            reader_worker = _ParallelWorker(reader_gen, input_queue)
            self._workers.append(reader_worker)

            # adding workers that will process the data
            for _ in range(self.worker_processes_number - 1):
                # since data-chunks will appear in the queue, making an iterable
                # object over it
                queue_iter = create_gen_from_queue(input_queue,
                                                   TERMINATION_TOKEN)
                data_itr = combine_steps_into_chain(data_producer=queue_iter,
                                                    processing_steps=self.steps,
                                                    error_on_invalid_chunk=self.error_on_invalid_chunk)
                proc_worker = _ParallelWorker(data_chunk_iter=data_itr,
                                              queue=output_queue)
                self._workers.append(proc_worker)
        else:
            term_tokens_expected = 1
            data_itr = combine_steps_into_chain(data_producer=reader_gen,
                                                processing_steps=self.steps,
                                                error_on_invalid_chunk=self.error_on_invalid_chunk)
            proc_worker = _ParallelWorker(data_chunk_iter=data_itr,
                                          queue=output_queue)
            self._workers.append(proc_worker)

        for pr in self._workers:
            pr.daemon = True
            pr.start()

        # terminate when all data is processed
        while term_tokens_received != term_tokens_expected:
            data_chunk = output_queue.get(timeout=self.timeout)
            if equal_to_constant(data_chunk, TERMINATION_TOKEN):
                term_tokens_received += 1
                continue
            yield data_chunk
        self._terminate_workers()

    def _terminate_workers(self):
        """Kills the workers."""
        for indx, pr in enumerate(self._workers):
            pr.terminate()
            pr.join()

    def __str__(self):
        """Converts the setup/configuration into a human readable string."""
        parent_title, parent_dict = self.get_title(), self.get_sign_attrs()

        chain = []
        if self.preprocessor:
            chain.append(self.preprocessor)
        chain += [self.reader] + self.steps

        children_titles = []
        children_dicts = []
        for step in chain:
            title, attrs = step.get_title(), step.get_sign_attrs()
            children_titles.append(title)
            children_dicts.append(attrs)

        str_setup = format_to_standard_msg_str(parent_title=parent_title,
                                               parent_dict=parent_dict,
                                               children_titles=children_titles,
                                               children_dicts=children_dicts)
        return str_setup

    def get_title(self):
        """Returns the formatted title of the object with a prefix if set."""
        base_title = "%s's SETUP" % self.__class__.__name__
        title = format_title(base_title.upper(),
                             name_prefix=self.name_prefix.upper(),
                             capitalize_prefix=False)
        return title

    def _adjust_scraper(self):
        """Changes the scraper's settings that collects signature attrs."""
        exl_attrs = ['steps', "preprocessor"]

        # exclude attrs which are not used for a specific architecture
        if self.worker_processes_number == 0:
            exl_attrs.append('input_buffer_size')
            exl_attrs.append('output_buffer_size')
        if self.worker_processes_number == 1:
            exl_attrs.append('input_buffer_size')

        self.scraper.excl_attr_names += exl_attrs
        self.scraper.scrape_obj_vals = False


class _ParallelWorker(Process):
    """Worker to execute data reading or processing on a separate process."""

    def __init__(self, data_chunk_iter, queue):
        super(_ParallelWorker, self).__init__()
        self._data_chunk_iterable = data_chunk_iter
        self._queue = queue

    def run(self):
        for data_chunk in self._data_chunk_iterable:
            self._queue.put(data_chunk)
        self._queue.put(TERMINATION_TOKEN)


def combine_steps_into_chain(data_producer, processing_steps,
                             error_on_invalid_chunk):
    """
    Chains processing components/processing_steps sequentially together.
    The chain can be iterated over to produce data-chunks.

    Can be used to produce data-chunks on demand (in the iteration loop).
    Alternatively, can be placed on a separate process that pre-populates a
    queue with processed data-chunks.

    :param data_producer: raw data-chunk producer. E.g. a reader object or just
                          an iterator over raw data-chunks.
    :param processing_steps: list of steps that need to be applied to data-chunks
                             to process them.
    :param error_on_invalid_chunk: self-explanatory.
    :return: generator.
    """
    prev_step = data_producer
    for i, new_step in enumerate(processing_steps):
        if isinstance(new_step, BaseGeneral):
            prev_step = new_step.iter(prev_step)
        else:
            vi = i == 0
            vo = not isinstance(new_step, BaseFormatter)
            prev_step = chain_two_steps(prev_step, new_step,
                                        error_on_invalid_chunk,
                                        validate_input=vi, validate_output=vo)
    return prev_step


def chain_two_steps(data_chunk_iterable, new_step, error_on_invalid_chunk,
                    validate_input=True, validate_output=True):
    """Combines two steps together and returns a generator of data-chunks."""
    for input_chunk in data_chunk_iterable:
        step_name = new_step.__class__.__name__
        if validate_input:
            suffix_msg = " Detected as input to '%s'." % step_name
            validate_chunk(input_chunk,
                           error_on_invalid_chunk=error_on_invalid_chunk,
                           suffix_msg=suffix_msg)

        output_chunk = new_step(input_chunk)

        if equal_to_constant(output_chunk, EMPTY_CHUNK):
            continue

        if validate_output:
            suffix_msg = " Detected as output from '%s'." % step_name
            validate_chunk(output_chunk,
                           error_on_invalid_chunk=error_on_invalid_chunk,
                           suffix_msg=suffix_msg)
        yield output_chunk


def validate_chunk(data_chunk, error_on_invalid_chunk=True, suffix_msg=""):
    """Wrapper over validation of chunks with different outcomes."""

    def _valid(dc):
        if not isinstance(dc, DataChunk):
            raise DataChunkError("The data-chunk is an invalid object.")
        data_chunk.validate()

    def _adjust_error(er, msg):
        er.args = tuple([er.args[0] + msg])
        # er.message += msg
        return er

    if error_on_invalid_chunk:
        try:
            _valid(data_chunk)
        except Exception as e:
            e = _adjust_error(e, suffix_msg)
            raise e
