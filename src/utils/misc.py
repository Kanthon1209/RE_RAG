import time
import logging

def timer(logger: logging.Logger):
    if logger is None:
        logging.basicConfig(level=logging.INFO)
        logger = logging.getLogger()
    def decorator(func):
        def wrapper(*args, **kwargs):
            start_time = time.time()
            rst = func(*args, **kwargs)
            end_time = time.time()
            logger.info(f"{func.__repr__()} cost {end_time - start_time:.3f} s")
            return rst
        return wrapper
    return decorator
