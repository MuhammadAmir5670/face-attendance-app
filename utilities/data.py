import os
from loguru import logger

from model.utilities.utils import get_folders, get_files, log_and_exit
from model.utilities.image import Image
from model.videoCapture import Stream, config
from collections import defaultdict


class Entity:
    def __init__(self, path, subject=None):
        self.path = path
        self.subject = subject
        self.name, self.label = self.verbose_name(self.path)
        self.images = Data.load(self.path, [Image]).get(Image.__name__)

    @staticmethod
    def isValid(path):
        directory = os.path.basename(path)
        if directory.startswith(config.StudentPrefix):
            return directory
        return False

    @staticmethod
    def verbose_name(path):
        directory = Entity.isValid(path)
        if directory:
            prefix, name, label = directory.split('_')
            return name, int(label)

    @property
    def processed(self):
        return hasattr(self, "encodings")

    def dir_name(self):
        return self.isValid(self.path)

    def faces(self, detector):
        for image in self.__iter__():
            face = detector.align_face(image_dimensions=96, image=image(),
                                       landmark_indices=detector.OUTER_EYES_AND_NOSE)
            if face is None:
                os.unlink(image.path)
                continue
            yield face

    def __str__(self):
        return self.name.title()

    def __repr__(self):
        return f'<Data.Entity: {self.name}>'

    def __iter__(self):
        self.current = 0
        return self

    def __next__(self):
        if self.current < len(self.images):
            image = self.images[self.current]
            self.current += 1
            return image
        else:
            raise StopIteration


class Data:
    def __init__(self, path):
        self.path = path
        self.name, self.assigned = self.verbose_name(self.path)
        self.entities = self.load(self.path, [Entity]).get(Entity.__name__)

    @staticmethod
    def verbose_name(path):
        directory = Data.isValid(path)
        if directory and directory.startswith(config.SUBJECT_PREFIX):
            elements = directory.split(config.SEPARATOR)
            return elements[-1], True
        else:
            return 'General', False

    @staticmethod
    def load(path, loaders: list = None, current=False):
        if not os.path.exists(path):
            log_and_exit("input path does not exists", logger.error)

        valid_loaders = (Data, Entity, Image, Stream)
        loaders = loaders if loaders else [Data]
        data = defaultdict(list)

        for loader in loaders:
            if loader not in valid_loaders:
                continue

            flag = loader.isValid(path)

            if flag:
                data[loader.__name__].append(loader(path))

            if not flag or current:
                files_dirs = get_files(path) if loader in (Image, Stream) else get_folders(path)

                for file_dir in files_dirs:
                    if not loader.isValid(os.path.join(path, file_dir)):
                        continue
                    data[loader.__name__].append(loader(os.path.join(path, file_dir)))

        return data

    def to_json(self):
        pass

    @staticmethod
    def isValid(path):
        directory = os.path.basename(path)
        if not Entity.isValid(path):
            return directory
        return False

    def preprocess(self, model, detector):
        for index, entity in enumerate(self.__iter__()):
            logger.info('preprocessing data for student label {}'.format(entity))
            entity.encodings = model.encoder(faces=entity.faces(detector=detector))

    def dir_name(self):
        return os.path.basename(self.path)

    def __str__(self):
        return self.name.title()

    def __repr__(self):
        return f'<Data.Subject: {self.name}>'

    def __iter__(self):
        self.current = 0
        return self

    def __next__(self):
        if self.current < len(self.entities):
            entity = self.entities[self.current]
            self.current += 1
            return entity
        else:
            raise StopIteration

