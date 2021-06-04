import os
import shutil
import sys
from loguru import logger
from model.utilities.config import config
import numpy as np
import pandas as pd


def log_and_exit(message, log):
    if message is not None:
        log(message)
    sys.exit()


def remove_directory(path):
    """
    deletes the directory with all its subdirectories and files
    :param path: path to the directory
    :return None:
    """
    if os.path.exists(path):
        logger.warning('deleting directory {} and all files within it'.format(os.path.basename(path)))
        shutil.rmtree(path)


def remove_files(path, directories=False):
    """
    Method for deleting all files in a directory
    :param directories: Boolean
    :param path: path to the directory which is to be empty
    :return None:
    """

    folders = get_folders(path)
    files = get_files(path)

    if directories:
        logger.warning('deleting all folders at {} path'.format(path))
        for folder in folders:
            remove_directory(os.path.join(path, folder))
    else:
        logger.warning('deleting all files at {} path'.format(path))
        for file in files:
            os.unlink(os.path.join(path, file))


def get_folders(path=None):
    assert path is not None, 'path is required'
    return next(os.walk(path))[1]


def get_files(path):
    return next(os.walk(path))[2]


def load_attendance(subject):
    attendance = '{}{}.csv'.format(subject, config.ATTENDANCE_FILE_SUFFIX)
    attendance = os.path.join(config.TRAINED_DATA, attendance)

    if os.path.isfile(attendance):
        data = pd.read_csv(attendance)
        labels = np.array(data.Labels)
        data.set_index('Labels', inplace=True)
        return data, labels
    return None, None


def save_attendance(subject, data, path=config.TRAINED_DATA,):
    attendance = '{}{}.csv'.format(subject, config.ATTENDANCE_FILE_SUFFIX)
    data.to_csv(os.path.join(path, attendance), index=True)
