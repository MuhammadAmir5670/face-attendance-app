import os
import pickle
from loguru import logger

from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import GridSearchCV
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from model.utilities.config import config


class Classifier:
    def __init__(self, data):
        self.data, self.classes = self.preprocess(data)

        # splitting data into dependent and independent
        self.x, self.y = self.data.iloc[:, :-1].values, self.data.iloc[:, -1].values

    @staticmethod
    def preprocess(data):
        # encoding categories to numbers
        encoder = LabelEncoder()
        data.iloc[:, -1] = encoder.fit_transform(data.iloc[:, -1])
        classes = len(encoder.classes_)
        return data, classes

    def train_svc(self):
        # defining parameter range
        param_grid = {'kernel': ('linear', 'rbf'),
                      'C': [1, 10],
                      'probability': [True]}
        model = self.fine_tune_model(SVC(), param_grid)
        model.fit(self.x, self.y)
        logger.info('optimal parameters for support vector classifier: {}'.format(model.best_estimator_))
        return model

    def train_knn(self):
        # defining parameter range
        param_grid = {'n_neighbors': range(5, 20),
                      'algorithm': ['auto', 'ball_tree', 'kd_tree', 'brute'],
                      'leaf_size': [30, 35, 40, 45, 50],
                      'p': [1, 2],
                      'weights': ['distance']
                      }
        model = self.fine_tune_model(KNeighborsClassifier(), param_grid)
        model.fit(self.x, self.y)
        print('optimal parameters for support vector classifier', model.best_params_)
        return model

    @staticmethod
    def fine_tune_model(model, param):
        grid = GridSearchCV(model, param_grid=param, refit=True, verbose=3, cv=2)
        return grid

    @staticmethod
    def load(path, subject):
        classifier = '{}{}.sav'.format(subject, config.CLASSIFIER_FILE_SUFFIX)
        classifier = os.path.join(path, classifier)
        if os.path.isfile(classifier):
            return pickle.load(open(classifier, 'rb'))

        return None


