import os
from datetime import datetime
import logging
from torch.utils.tensorboard import SummaryWriter


def make_file(log_dir, file_name):
    f = open(os.path.join(log_dir, file_name), "w+")
    f.close()
    return os.path.join(log_dir, file_name)


def make_writer(log_dir):
    writer = SummaryWriter(log_dir=log_dir)
    return writer

def log_initilize(log_path):
    logger = logging.getLogger()
    logger.setLevel(logging.DEBUG)

    # # create console handler and set level to info
    # handler = logging.StreamHandler()
    # handler.setLevel(logging.INFO)
    # formatter = logging.Formatter("%(message)s")
    # handler.setFormatter(formatter)
    # handler.terminator = "\n"
    # logger.addHandler(handler)

    # create error file handler and set level to error
    handler = logging.FileHandler(log_path, "a", encoding=None, delay="true")
    handler.setLevel(logging.INFO)
    formatter = logging.Formatter("%(message)s")
    handler.setFormatter(formatter)
    handler.terminator = "\n"
    logger.addHandler(handler)
