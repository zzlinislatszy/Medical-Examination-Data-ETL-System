import os
import json
import time


# 印出執行時間
def log_execution_time(func):
    def wrapper(*args, **kwargs):
        start_time = time.time()  # Start time before function call
        result = func(*args, **kwargs)  # Call the actual function
        end_time = time.time()  # End time after function call
        execution_time = end_time - start_time  # Calculate execution time
        print(f"{func.__name__} executed in {execution_time:.6f} seconds")
        return result
    return wrapper
