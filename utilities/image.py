import cv2
from os.path import basename, isfile, join

from loguru import logger
from filetype import guess

from skimage import exposure
from skimage.feature import hog


class Image:
    """
    Custom Image class to store more information about the Image
    wrapper around the openCV image object
    """

    def __init__(self, path, entity=None):
        self.path = path
        self.name = self.verbose_name(self.path)
        self.entity = entity

    @staticmethod
    def isValid(path):
        if isfile(path):
            kind = guess(path)
            if kind is not None and kind.mime.startswith("image"):
                return basename(path)
        return False

    @staticmethod
    def verbose_name(path):
        return Image.isValid(path)

    def __call__(self):
        """
        will load the image and return
        :return:
        """
        logger.info('loading image: {}'.format(self.name))
        image = cv2.imread(self.path)
        return image

    def __str__(self):
        return self.name

    def __repr__(self):
        return f'<Data.Image: path={self.path}>'


# OpenCv Utilities
def normalize_histogram(image, grid=16):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    normalizer = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(grid, grid))
    normalized_image = normalizer.apply(image)
    return normalized_image


def display_image(image, label='image'):
    """
    Method for displaying the image
    :param image: Image Object
    :param label: Text to be displayed on the window
    :return None:
    """
    # converting image to RGB before displaying
    # image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)

    # displaying the image
    cv2.imshow(label, image)

    # waiting util the user does not press key => 0
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def save_image(image, name, path, converter=None):
    """
    Method for saving the image
    :param converter:
    :param image: object of image that is to be saved
    :param name: name by the image should be saved
    :param path: path to the directory where image should be saved
    :return None:
    """
    # if converter is defined
    if converter is not None:
        logger.debug('converting image')
        # convert the image channels
        image = cv2.cvtColor(image, converter)
    path = join(path, name)
    cv2.imwrite(path, image)


# Utilities Used by Detector.py
def rect_to_bounding_box(rect):
    """
    Method for converting the dlib's rect object
    to standard --coordinates of (x1, y1) and (x2, y2)
    :param rect:
    :return tuple: (x1, y1, x2, y2)
    """
    x = rect.left()
    y = rect.top()
    w = rect.right() - x
    h = rect.bottom() - y

    return x, y, w, h


def draw_text(image, text, face_location, color=(0, 0, 255), rect=True):
    """
    function to draw text on give image starting from
    passed (x, y) coordinates.
    :param color:
    :param rect:
    :param image:
    :param text:
    :param face_location:
    :return:
    """

    if rect:
        x1, y1, x2, y2 = rect_to_bounding_box(face_location)
    else:
        x1, y1, x2, y2 = tuple(face_location)

    cv2.putText(image, text, (x1, y1), cv2.FONT_HERSHEY_PLAIN, 1.5, color, 2)


def draw_rectangles(image, face_locations, color=(0, 0, 255), thickness=5, rect=True):
    """
    Method for drawing Rectangles around the detected face locations
    :param rect:
    :param image: image-object
    :param face_locations: list of tuples
    :param color: tuple of three elements
    :param thickness: integer value
    :return image: a new image with rectangles drawn on it
    """
    logger.info('Drawing rectangles around detected faces')
    if rect:
        face_locations = list(map(lambda location: rect_to_bounding_box(location), face_locations))

    for (x1, y1, x2, y2) in face_locations:
        cv2.rectangle(image, (x1, y1), (x1 + x2, y1 + y2), color, thickness)
    return image


def draw_landmarks(image, landmarks=None, color=(255, 0, 0)):
    for x, y in landmarks:
        cv2.circle(image, (x, y), 1, color, 2)


def image_to_hog(image):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    fd, hog_image = hog(image, orientations=8, pixels_per_cell=(16, 16), cells_per_block=(1, 1), visualize=True)
    hog_image_rescaled = exposure.rescale_intensity(hog_image, in_range=(0, 10))
    return hog_image_rescaled
