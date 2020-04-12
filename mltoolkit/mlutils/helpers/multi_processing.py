from .general import equal_to_constant
import multiprocessing.queues
import os


def create_gen_from_queue(queue, term_token):
    """
    Creates a iterable generator of a multiprocessing queue, which is
    stopped when the "poison pill" is found in the queue.
    :param queue: multiprocessing Queue object
    :param term_token: poison pill indicating that it should not be expected
                       to find more data in the queue.
    :return: generator
    """
    if not isinstance(queue, multiprocessing.queues.Queue):
        raise ValueError("Please provide a valid multiprocessing queue.")
    while True:
        input_data_chunk = queue.get()
        if equal_to_constant(input_data_chunk, term_token):
            # put it back to the queue to let other processes that feed
            # from the same one to know that they should also break
            queue.put(term_token)
            break
        else:
            yield input_data_chunk


def get_os_seed():
    """Returns a seed number based on OS random numbers."""
    return ord(os.urandom(1))
