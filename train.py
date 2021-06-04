import os
import pickle
import argparse
import pandas as pd
from loguru import logger

from model.utilities.classifier import Classifier
from preprocess import Preprocessor
from utilities.utils import log_and_exit, save_attendance
from model.utilities.config import config


def train(data, subject=None):
    # saving the csv file for marking attendance
    labels = pd.unique(data.Labels)
    labels = pd.DataFrame(labels, columns=['Labels'])
    labels.set_index('Labels', inplace=True)

    # Save the csv file for taking attendance
    save_attendance(subject=subject, data=labels)

    # Training the classifier
    classifier = Classifier(data)
    print(config.CLASSIFIER_FILE_SUFFIX)
    # SVC classification
    svc = classifier.train_svc()
    pickle.dump(svc, open(os.path.join(config.TRAINED_DATA, '{}{}.sav'.format(subject, config.CLASSIFIER_FILE_SUFFIX)), 'wb'))


def main(arguments):
    # check if the given path is given or not
    if not os.path.isdir(arguments.input):
        log_and_exit("invalid input path is given: directory does not exist", logger.error)

    if not os.path.isdir(arguments.output):
        log_and_exit("invalid output path is given: directory does not exist", logger.error)

    # initiate data processor
    preprocessor = Preprocessor(path=arguments.input)
    preprocessor.preprocess()

    for subject in preprocessor.data:
        data = Preprocessor.to_dataframe(subject)
        train(data=data, subject=subject.name)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Encoding Faces and Training classifier on Encodings')
    parser.add_argument('-i', '--input', help='path to the input folder or subject folder',
                        metavar='', type=str, default=os.path.join(config.TRAINING_DATA))
    parser.add_argument('-o', '--output', help='path to the folder where trained results are to be saved',
                        metavar='', type=str, default=os.path.join(config.TRAINED_DATA))

    args = parser.parse_args()
    main(args)
