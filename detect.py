# import required libraries
import cv2
from os import unlink
import dlib
import numpy as np
import argparse
from loguru import logger

from model.utilities.utils import remove_files
from model.utilities.image import save_image, draw_rectangles, normalize_histogram
from model.utilities.data import Data, Image
from model.utilities.defaults import MIN_MAX_TEMPLATE
from model.utilities.config import config


class FaceDetector:
    """
    Use `dlib's landmark estimation to align faces.
    The alignment preprocess faces for input into a neural network.
    Faces are resized to the same size (such as 96x96) and transformed
    to make landmarks (such as the eyes and nose) appear at the same
    location on every image.
    Normalized landmarks:
    .. image:: ../images/dlib-landmark-mean.png
    """

    #: Landmark indices.
    INNER_EYES_AND_BOTTOM_LIP = [39, 42, 57]
    OUTER_EYES_AND_NOSE = [36, 45, 33]

    def __init__(self, face_predictor):
        """
        Instantiate an 'AlignDlib' object.
        :param face_predictor: The path to dlib's
        :type face_predictor: str
        """
        assert face_predictor is not None

        self.detector = dlib.get_frontal_face_detector()
        self.predictor = dlib.shape_predictor(face_predictor)

    def get_all_faces(self, image):
        """
        Find all face bounding boxes in an image.
        :param image: RGB image to process. Shape: (height, width, 3)
        :type image: numpy.ndarray
        :return: All face bounding boxes in an image.
        :rtype: dlib.rectangles
        """
        assert image is not None
        image = normalize_histogram(image)
        try:
            return self.detector(image, 1)
        except Exception as e:
            logger.warning("Warning: {}".format(e))
            # In rare cases, exceptions are thrown.
            return []

    def get_face(self, image, skip_multi=False):
        """
        Find the largest face bounding box in an image.
        :param image: RGB image to process. Shape: (height, width, 3)
        :type image: numpy.ndarray
        :param skip_multi: Skip image if more than one face detected.
        :type skip_multi: bool
        :return: The largest face bounding box in an image, or None.
        :rtype: dlib.rectangle
        """
        assert image is not None

        faces = self.get_all_faces(image)
        if (not skip_multi and len(faces) > 0) or len(faces) == 1:
            return max(faces, key=lambda rect: rect.width() * rect.height())
        else:
            return None

    def find_landmarks(self, image, bounding_box):
        """
        Find the landmarks of a face.
        :param image: RGB image to process. Shape: (height, width, 3)
        :type image: numpy.ndarray
        :param bounding_box: Bounding box around the face to find landmarks for.
        :type bounding_box: dlib.rectangle
        :return: Detected landmark locations.
        :rtype: list of (x,y) tuples
        """
        assert image is not None
        assert bounding_box is not None

        points = self.predictor(image, bounding_box)

        # converting points object to list of (x,y) - coordinates
        coordinates = list(map(lambda p: (p.x, p.y), points.parts()))
        return coordinates

    def align_face(self, image_dimensions, image, bounding_box=None,
                   landmarks=None, landmark_indices=INNER_EYES_AND_BOTTOM_LIP,
                   skip_multi=False):
        """
        align(imgDim, rgbImg, bb=None, landmarks=None, landmarkIndices=INNER_EYES_AND_BOTTOM_LIP)
        Transform and align a face in an image.
        :param image_dimensions: The edge length in pixels of the square the image is resized to.
        :type image_dimensions: int
        :param image: RGB image to process. Shape: (height, width, 3)
        :type image: numpy.ndarray
        :param bounding_box: Bounding box around the face to align. \
                   Defaults to the largest face.
        :type bounding_box: dlib.rectangle
        :param landmarks: Detected landmark locations. \
                          Landmarks found on `bb` if not provided.
        :type landmarks: list of (x,y) tuples
        :param landmark_indices: The indices to transform to.
        :type landmark_indices: list of ints
        :param skip_multi: Skip image if more than one face detected.
        :type skip_multi: bool
        :return: The aligned RGB image. Shape: (imgDim, imgDim, 3)
        :rtype: numpy.ndarray
        """
        assert image_dimensions is not None
        assert image is not None
        assert landmark_indices is not None

        if bounding_box is None:
            bounding_box = self.get_face(image, skip_multi)
            if bounding_box is None:
                return

        if landmarks is None:
            landmarks = self.find_landmarks(image, bounding_box)

        np_landmarks = np.float32(landmarks)
        np_landmark_indices = np.array(landmark_indices)

        H = cv2.getAffineTransform(np_landmarks[np_landmark_indices],
                                   image_dimensions * MIN_MAX_TEMPLATE[np_landmark_indices])
        thumbnail = cv2.warpAffine(image, H, (image_dimensions, image_dimensions))

        return thumbnail

    def align_all_faces(self, image, bounding_boxes=None):
        if bounding_boxes is None:
            bounding_boxes = self.get_all_faces(image)

        faces = []
        for box in bounding_boxes:
            faces.append(self.align_face(image_dimensions=96, image=image, bounding_box=box, landmarks=None,
                                         landmark_indices=self.OUTER_EYES_AND_NOSE))

        return faces


def main(arguments):
    """
    The module will start its execution from this method : main
    if it is executed directly
    :return:
    """

    remove_files(arguments.output)
    data = Data.load(arguments.input, loaders=[Image]).get(Image.__name__)

    logger.info('input data: {}'.format(list(map(lambda obj: str(obj), data))))

    logger.info('initializing face detector and face landmarks predictor')
    detector = FaceDetector(config.DETECTOR)

    for file in data:
        image = file()
        faces = detector.get_all_faces(image=image)
        draw_rectangles(image=image, face_locations=faces)

        if config.SAVE:
            save_image(name=str(file), image=image, path=arguments.output)

        if config.REMOVE:
            unlink(file.path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='detects faces in the images')
    parser.add_argument('-i', '--input', help='path to the input folder or image',
                        metavar='', type=str, default='test-data/test-image-4.jpg')
    parser.add_argument('-o', '--output', help='path to the output folder',
                        metavar='', type=str, default='output/detector')

    args = parser.parse_args(['--image'])
    main(args)


# Future Work: Implement Retina Face or MTCNN for detection

