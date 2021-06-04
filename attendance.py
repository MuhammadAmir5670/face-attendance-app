import argparse
from datetime import date
import os
import numpy as np
import pandas as pd

from model.utilities.config import config
from model.utilities.data import Data, Image, Stream
from model.inference import Predict
from collections import defaultdict


class Attendance:
    MODE = "image/video"

    def __init__(self, subject, threshold, model=None):
        self.model = model if isinstance(model, Predict) else Predict()
        self.subject = subject
        self.threshold = threshold
        self.records = []
        self.attendance = {}
        self.attendance_sheet, self.enrolled_std_labels = self.load()

    def record(self, path):
        if path == 0:
            self.camera()
            return

        files = []
        files.extend(Data.load(path, loaders=[Image]).get(Image.__name__, []))
        files.extend(Data.load(path, loaders=[Stream]).get(Stream.__name__, []))

        for file in files:
            recorder = Recorder(self, file=file)
            recorder.start()
            self.records.append(recorder)

        self.attendance.update(Recorder.merge(self.records))

    def camera(self):
        camera = Stream(path=0, display=True)
        recorder = Recorder(self, file=camera)
        recorder.start()
        self.attendance.update(recorder.entities)
        self.records.append(recorder)

    def mark(self):
        today = date.today()
        self.attendance_sheet[str(today)] = '-'

        for entity, probability in self.attendance.items():
            if entity != 'unknown':
                self.attendance_sheet.loc[entity] = 'Present'

        self.save(data=self.attendance_sheet)

    def presentees(self):
        return [label for label, probability in self.attendance]

    def absentees(self):
        absent_stds = set(self.enrolled_std_labels) - set(self.presentees())
        return list(absent_stds)

    def load(self):
        attendance = '{}{}.csv'.format(self.subject, config.ATTENDANCE_FILE_SUFFIX)
        attendance = os.path.join(config.TRAINED_DATA, attendance)
        if os.path.isfile(attendance):
            data = pd.read_csv(attendance)
            labels = np.array(data.Labels)
            data.set_index('Labels', inplace=True)
            return data, labels
        return None, None

    def save(self, data):
        attendance = '{}{}.csv'.format(self.subject, config.ATTENDANCE_FILE_SUFFIX)
        data.to_csv(os.path.join(config.TRAINED_DATA, attendance), index=True)

    @staticmethod
    def exist(subject):
        path = os.path.join(config.TRAINED_DATA, f"{subject}{config.CLASSIFIER_FILE_SUFFIX}.sav")
        if os.path.isfile(path):
            return subject
        return False

    def __str__(self):
        return f"<Attendance: {self.subject}>"


class Recorder(Attendance):
    def __init__(self, parent=None, **kwargs):
        if isinstance(parent, Attendance):
            self.__dict__ = parent.__dict__.copy()
        else:
            parent_params = {}
            if kwargs.get("model"):
                parent_params["model"] = kwargs.get("model")
            if kwargs.get("subject"):
                parent_params["subject"] = kwargs.get("subject")

            super(Recorder, self).__init__(**parent_params)

        self.file = kwargs.get("file")
        self.entities = {}

    @staticmethod
    def merge(records):
        merged = defaultdict(list)

        for record in records:
            for entity, probability in record.entities.items():
                merged[entity].append(probability)

        for entity, trust_vector in merged.items():
            array = np.array(trust_vector)
            averaged = np.average(array)
            merged[entity] = averaged

        return merged

    def start(self):
        if isinstance(self.file, Stream):
            self.video()
        if isinstance(self.file, Image):
            self.image()

    def video(self):
        entities = self.file(model=self.model, subject=self.subject, threshold=self.threshold)
        for label, tracker in entities.items():
            self.entities.setdefault(label, tracker.probability)

    def image(self):
        image = self.file()
        labels, trust_vector, faces = self.model(image=image, subject=self.subject, threshold=self.threshold)

        for label, probability, face in zip(labels, trust_vector, faces):
            self.entities.setdefault(label, probability)


def main(arguments):
    attendance = Attendance(subject=arguments.subject, threshold=27)
    attendance.record(path=0)
    attendance.mark()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Mark Attendance using the faces in the images')
    parser.add_argument('-i', '--input', help='path to the image or video', metavar='', type=str,
                        default=os.path.join(config.BASE_DIR, 'test-data'))
    parser.add_argument('-c', '--subject', help='path to the class folder which attendance is to be taken',
                        metavar='', type=str, default='General')

    args = parser.parse_args()
    main(args)
