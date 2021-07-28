import time
from functools import wraps


def measure_time(func):
    @wraps(func)
    def time_it(*args, **kwargs):
        print(f'Testing "{func.__name__}"')
        start_timer = time.perf_counter()
        try:
            return func(*args, **kwargs)
        finally:
            end_timer = time.perf_counter()
            print(f"Execution time: {end_timer - start_timer} sec")
    return time_it
