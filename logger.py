import os
import torch
from datetime import datetime


class Logger:
    def __init__(self, output_dir, log_time=True, log_file_name='logs.txt'):
        self.log_file_path = os.path.join(output_dir, log_file_name)
        self.log_time = log_time

    def log(self, text, print_to_console=True):
        if self.log_time:
            time_string = f'{datetime.now().strftime("%d.%m.%Y %H:%M:%S")}\t'
        else:
            time_string = ''
        line_to_write = f'{time_string}{text}\n'
        with open(self.log_file_path, 'a') as file:
            file.write(line_to_write)
        if print_to_console:
            print(line_to_write, end='')