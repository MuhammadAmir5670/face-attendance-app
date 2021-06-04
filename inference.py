import numpy as np
import argparse
from loguru import logger
from collections import defaultdict
import os

from model.faceNet import FaceNet
from model.detect import FaceDetector
from model.utilities.utils import remove_files, load_attendance
from model.utilities.image import save_image, draw_rectangles, draw_text
from model.utilities.data import Data, Image
from model.utilities.classifier import Classifier
from model.utilities.config import config


# this function recognizes the person in image passed
# and draws a rectangle around detected face with name of the
# subject
class Predict:
    def __init__(self):
        self.detector = FaceDetector(config.DETECTOR)
        self.model = FaceNet()

    def __call__(self, image, subject, threshold, bounding_boxes=None, faces=None):
        encodings, bounding_boxes = self.predict(image=image, bounding_boxes=bounding_boxes, faces=faces)
        labels, trust_vector = self.recognize(subject=subject, threshold=threshold, encodings=encodings)
        return self.remove_duplicates(labels, trust_vector, bounding_boxes)

    def recognize(self, subject, threshold, encodings=None, image=None):
        # ------STEP-5--------
        # identify all the detected faces
        # by comparing the 128 embeddings of each face
        # with face already present in the database

        if encodings is None and image is not None:
            encodings, bounding_boxes = self.predict(image=image)

        logger.info('recognizing the detected faces in image')

        classifier = Classifier.load(os.path.join(config.TRAINED_DATA), subject)
        _, names = load_attendance(subject=subject)

        labels, trust_vector = [], []
        for encoding in encodings:
            # recognize the person using the pretrained classifier
            # reshape the encoding vector w.r.t classifier's input
            encoding = encoding.reshape(1, 128)
            label = classifier.predict(encoding)

            # predicting the probability of recognition
            vector = classifier.predict_proba(encoding)[0]  # get the probability vector
            index = np.argmax(vector)  # get index of highest probability in the vector
            probability = vector[index] * 100

            # recognition is correct if the probability is above a certain threshold
            label = names[label[0]] if probability > threshold else 'unknown'
            labels.append(label)
            trust_vector.append(probability)

            logger.info('{}\'s face detected in image'.format(label))

        return labels, trust_vector

    def predict(self, image, bounding_boxes=None, faces=None):
        # ------STEP-1--------
        # detect face from the image
        if bounding_boxes is None and faces is None:
            bounding_boxes = self.detector.get_all_faces(image=image)

        # ------STEP-2--------
        # align each of the detected face
        if faces is None:
            faces = self.detector.align_all_faces(image, bounding_boxes=bounding_boxes)

        # ------STEP-3--------
        # create a feature vector
        # to store the 128 embeddings of the n faces detected in the image
        logger.info('creating feature vector for the detected faces')

        # ------STEP-4--------
        # encode each detected face into 128 features
        logger.info('encoding detected faces')
        encodings = self.model.encoder(faces=faces)

        return encodings, bounding_boxes

    @staticmethod
    def remove_duplicates(labels, trust_vector, faces):
        duplicates = defaultdict(list)
        unique = defaultdict(list)
        # create a mapping of values and indexes
        for index, label in enumerate(labels):
            duplicates[label].append(index)

        # filter out the unique labels and indices
        indices = tuple(indices[0] for key, indices in duplicates.items() if len(indices) == 1)
        for index in indices:
            unique["labels"].append(labels[index])
            unique["trust_vector"].append(trust_vector[index])
            unique["faces"].append(faces[index])

        # filter out the duplicate labels and indices
        duplicates = {key: value for key, value in duplicates.items() if len(value) > 1}

        for label in duplicates:
            maximum = max(duplicates[label], key=lambda i: trust_vector[i])
            for index in duplicates[label]:
                unique['labels'].append(labels[index] if index == maximum else "unknown")
                unique["trust_vector"].append(trust_vector[index])
                unique["faces"].append(faces[index])

        return unique["labels"], unique["trust_vector"], unique["faces"]


def main(arguments):
    subject, is_subject = Data.verbose_name(arguments.subject)

    remove_files(arguments.output)

    data = Data.load(arguments.input, loaders=[Image]).get(Image.__name__)

    predictor = Predict()

    for file in data:
        logger.warning('======================================================================')
        image = file()
        labels, trust_vector, faces = predictor(image=image, subject=subject, threshold=arguments.threshold)
        for label, face, probability in zip(labels, faces, trust_vector):

            draw_rectangles(image, [face])
            draw_text(image, f"{label} - {probability}", face)

        logger.info('saving image {}'.format(file))
        save_image(image, 'output-{}'.format(file), arguments.output)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='predict faces appear in the images')
    parser.add_argument('-i', '--input', help='path to the input folder or image',
                        metavar='', type=str, default=os.path.join(config.BASE_DIR, "test-data"))
    parser.add_argument('-o', '--output', help='path to the output folder',
                        metavar='', type=str, default=os.path.join(config.OUTPUT, "predictor"))
    parser.add_argument('-t', '--trained-data', help='path to the folder containing trained-data data',
                        metavar='', type=str, default=config.TRAINED_DATA)
    parser.add_argument('-c', '--subject', help='path to the class folder which attendance is to be taken',
                        metavar='', type=str, default='General')
    parser.add_argument('-p', '--threshold', help='threshold value for recognizing the faces',
                        metavar='', type=int, default=27)

    args = parser.parse_args()
    main(args)
