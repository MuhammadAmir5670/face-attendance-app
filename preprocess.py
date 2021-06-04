import os
from loguru import logger
import argparse

from model.detect import FaceDetector
from model.utilities.data import Data
from model.faceNet import FaceNet
from model.utilities.config import config

from numpy import array
from pandas import DataFrame


class Preprocessor:
    def __init__(self, path):
        self.detector = FaceDetector(config.DETECTOR)
        self.model = FaceNet()
        self.path = path
        self.data = []

    def preprocess(self, path=None):
        """
        function for reading all person's trained-data images, detect face from each image
        :param path:
        :return:
        """
        path = path if path else self.path
        data = Data.load(path=path).get(Data.__name__)
        for subject in data:
            subject.preprocess(model=self.model, detector=self.detector)
            self.data.append(subject)

        return self.data

    @staticmethod
    def to_dataframe(data: Data, label="name") -> DataFrame:
        """
        :rtype: DataFrame
        """
        encodings = []
        labels = []
        data.entities = sorted(data.entities, key=lambda item: getattr(item, label))
        for entity in data.entities:
            if entity.processed:
                for encoding in getattr(entity, "encodings"):
                    encodings.append(encoding)
                    labels.append(getattr(entity, label))

        encodings = array(encodings)
        labels = array(labels)
        data = DataFrame(encodings)
        data["Labels"] = labels
        return data


def main(arguments):
    preprocessor = Preprocessor()
    preprocessor.preprocess()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Encoding Faces and Training classifier on Encodings')
    parser.add_argument('-i', '--input', help='path to the input folder or subject folder',
                        metavar='', type=str, default=config.TRAINING_DATA)
    parser.add_argument('-o', '--output', help='path to the folder where trained results are to be saved',
                        metavar='', type=str, default=os.path.join(config.OUTPUT, 'trained-data'))

    args = parser.parse_args()
    main(args)
