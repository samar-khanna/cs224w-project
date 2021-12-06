import os

def print_and_log(log_file, message):
    print(message)
    log_file.write(message + '\n')