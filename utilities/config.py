from os.path import dirname, join, abspath


class Config:
    BASE_DIR = abspath(dirname(dirname(__file__)))

    # Path to the pre-trained model weights
    MODEL = join(BASE_DIR, 'models/open_face.h5')

    # Path to the pre-trained model weights
    DETECTOR = join(BASE_DIR, 'models/landmarks.dat')

    # path to the output folder
    OUTPUT = join(BASE_DIR, 'output')

    # path to folder where the resultant files are to be saved after training
    TRAINED_DATA = join(OUTPUT, "trained-data")

    # path to folder from the data is to be taken for training
    TRAINING_DATA = join(BASE_DIR, "training-data")

    # save the files after training
    SAVE = True

    # remove files after training
    REMOVE = False

    # number of consecutive frames in which the entity should appear to be mark present
    MAX_FRAME = 5

    # Name Separator
    SEPARATOR = "__"

    # attendance csv file suffix
    ATTENDANCE_FILE_SUFFIX = SEPARATOR + "Attendance"

    # classifier bot file suffix
    CLASSIFIER_FILE_SUFFIX = SEPARATOR + "svc"

    # subject prefix
    SUBJECT_PREFIX = "class" + SEPARATOR

    StudentPrefix = "student_"

    GLOBAL = "unspecific" + SEPARATOR

    DATE_FORMAT = "%d-%m-%y | %H:%M:%S"


config = Config()
