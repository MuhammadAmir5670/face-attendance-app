import time
import cv2
import numpy as np
from os.path import basename, isfile
from model.utilities.config import config
from model.utilities.image import draw_rectangles, draw_text
from filetype import guess


class Stream:

    def __init__(self, path, display=False):
        self.path = path
        self.stream = cv2.VideoCapture(self.path)
        self.number_of_frames = int(self.stream.get(7))
        self.frame_per_sec = int(self.stream.get(5))
        self.current_index = 0
        self.display = display
        self.entities = {}

    @staticmethod
    def isValid(path):
        if isfile(path):
            kind = guess(path)
            if kind is not None and kind.mime.startswith("video"):
                return basename(path)
        return False

    @staticmethod
    def verbose_name(path):
        return Stream.isValid(path)

    def __call__(self, model, subject, threshold):
        for count, frame in self.__iter__():
            image = frame.track(*(model(image=frame.image, subject=subject, threshold=threshold)))

            if self.display:
                cv2.imshow('frame', image)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                self.stream.release()
                cv2.destroyAllWindows()
                break

        return self.entities

    def capture(self, detector=None):
        for count, frame in self.__iter__():
            image = frame.image
            if detector:
                faces = detector.get_all_faces(image=image)
                draw_rectangles(image, faces)

            if self.display:
                cv2.imshow('frame', image)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                self.stream.release()
                cv2.destroyAllWindows()
                break

    def __len__(self):
        return self.number_of_frames

    def __iter__(self):
        self.start_time = time.time()
        self.current_index = 0
        if not self.path == 0:
            self.stream = cv2.VideoCapture(self.path)

        self.number_of_frames = int(self.stream.get(7))
        return self

    def __next__(self):
        self.current_index += 1
        ret, frame = self.stream.read()  # ret is false at EOF
        if ret is False:
            self.stream.release()  # When everything done, release the video capture object
            cv2.destroyAllWindows()  # Closes all the frames
            self.current_index = None
            self.stream = None
            raise StopIteration  # stop the loop

        elif ret is True:
            # cv2 opens in bgr mode and needs to be converted to RGB
            return self.current_index, Frame(parent=self, image=frame, timestamp=self.stream.get(0))


class Frame(Stream):

    def __init__(self, image, timestamp, parent=None, path=None):
        if isinstance(parent, Stream):
            self.__dict__ = parent.__dict__.copy()
        else:
            super(Frame, self).__init__(path=path)
        self.image = image
        self.timestamp = timestamp
        self.labels = []
        self.trust_vector = []
        self.faces = []

    def __str__(self):
        return "timestamp: {}, entities: {}".format(self.timestamp / 1000, self.labels)

    def __repr__(self):
        return "timestamp: {}, entities: {}".format(self.timestamp / 1000, self.labels)

    def track(self, labels, trust_vector, faces):
        image = self.image.copy()
        self.labels.extend(labels)
        self.trust_vector.extend(trust_vector)
        self.faces.extend(faces)

        for label, probability, face in zip(labels, trust_vector, faces):
            student = self.entities.setdefault(label, Tracker(max_frames=5))
            student.present_in_frame(probability)
            student.frames = self

            draw_rectangles(image, [face])
            draw_text(image=image, text=label, face_location=face)

            missing_entities = self.missing_in_current_frame()

            for entity in missing_entities:
                # reduce count by one if a person is missing after appearing in the previous frame
                entity.not_present_in_frame()

        else:
            if self.entities.get("unknown"):
                self.entities.pop("unknown")

        return image

    def missing_in_current_frame(self):
        prev_detected = set(self.entities.keys())
        in_curr_frame = set(self.labels)
        not_detected = prev_detected - in_curr_frame
        return [self.entities[entity] for entity in not_detected]


class Tracker:

    def __init__(self, max_frames=config.MAX_FRAME):
        self.detected = True
        self.count = 0
        self.present = False
        self.frame_threshold = max_frames
        self.__trust_vector = []
        self.__probability = None
        self.__frames = []

    def __str__(self):
        return "count: {}, present: {}, probability: {}".format(self.count, self.present, self.probability)

    def __repr__(self):
        return "count: {}, present: {}, probability: {}".format(self.count, self.present, self.probability)

    @property
    def trust_vector(self):
        return self.__trust_vector

    @trust_vector.setter
    def trust_vector(self, value):
        if hasattr(value, "__iter__"):
            self.__trust_vector.extend([i for i in value if isinstance(i, int) or isinstance(i, float)])
        else:
            self.__trust_vector.append(value)

    @property
    def probability(self):
        if self.__probability:
            return self.__probability
        else:
            self.align_trust_vector()
            return self.probability

    @probability.setter
    def probability(self, value):
        if self.__probability:
            self.align_trust_vector(value)
        else:
            self.__probability = value

    @property
    def frames(self):
        return self.__frames

    @frames.setter
    def frames(self, value):
        if isinstance(value, Frame):
            self.__frames.append(value)
        elif hasattr(value, "__iter__"):
            self.__frames.extend([obj for obj in value if isinstance(obj, Frame)])
        else:
            raise ValueError("value should be an instance of Frame class")

    def align_trust_vector(self, *args):
        if self.trust_vector:
            self.trust_vector = args
            array = np.array(self.trust_vector)
            averaged = np.average(array)
            self.__probability = averaged

    def present_in_frame(self, probability):
        self.trust_vector = probability
        if self.count < self.frame_threshold:
            self.count += 1
            self.detected = True
        else:
            self.present = True

    def not_present_in_frame(self):
        # self.detected => entity was present in last frame
        # self.present => and it has not been marked present
        if self.detected and not self.present:
            self.count -= 1
            self.detected = False


if __name__ == "__main__":
    stream = Stream(0)

    for frame_number, frame_obj in stream:

        cv2.imshow('frame', frame_obj.image)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
