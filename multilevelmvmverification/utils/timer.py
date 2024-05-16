import time
from multilevelmvmverification.utils.logger import setup_logger

perf_logger = setup_logger('performance')


def time_method(method, params, f_y: callable = None):
    start = time.time()
    result = method(params, f_y)
    end = time.time()

    perf_logger.info(f"{method.__name__} took {end - start} seconds")
    return result
