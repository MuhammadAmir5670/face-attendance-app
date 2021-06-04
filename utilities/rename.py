import os
import argparse
from sort import getFiles


def TestData(path='test-data'):
    pattern = r'test-image-\d+'
    print('=== renaming test files')
    to_renamed, already_renamed = getFiles(path, pattern)
    print('=== files to be renamed \n', to_renamed)
    print('=== files already renamed \n', already_renamed)
    already_renamed = len(already_renamed)
    print('=== {} files are to be renamed'.format(len(to_renamed)))
    print('=== {} files are already renamed'.format(already_renamed))

    for count, file in enumerate(to_renamed):
        file_name, file_ext = os.path.splitext(file)
        source = os.path.join(path, file)
        destination = os.path.join(path, 'test-image-{}{}'.format(count + already_renamed, file_ext))
        print(source, destination, sep=' => ')
        os.rename(source, destination)
    print('=== test files renamed')


def TrainingData(path='trained-data-data'):
    rooms = os.listdir(path)
    print('=== classes found are {}'.format(rooms))
    for room in rooms:
        students = os.listdir(os.path.join(path, room))
        print('=== {} has {} students in it'.format(room, students))
        for student in students:
            directory = os.path.join(path, room, student)

            # ignore folders that does represent an entity
            if not student.startswith('student_'):
                continue

            # ignore files
            if not os.path.isdir(directory):
                continue
            print()
            # pattern for the file name
            pattern = r'image-\d+'
            print('=== renaming trained-data Files for student {}'.format(student))
            to_renamed, already_renamed = getFiles(directory, pattern)
            print('=== files to be renamed {}'.format(to_renamed))
            print('=== files already renamed {}'.format(already_renamed))
            already_renamed = len(already_renamed)
            print('=== {} files are to be renamed'.format(len(to_renamed)))
            print('=== {} files are already renamed'.format(already_renamed))

            for count, file in enumerate(to_renamed):
                # get the file name and its extension
                file_name, file_ext = os.path.splitext(file)
                # get the path to the file
                source = os.path.join(path, room, student, file)
                # get the path to the destination with new file name
                destination = os.path.join(path, room, student, 'image-{}{}'.format(count, file_ext))
                print(source, destination, sep=' => ')

                # rename the file from source to destination
                os.rename(source, destination)


def rename(config):
    if config.test:
        TestData()
    if config.train:
        TrainingData()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='rename the test and trained-data files')
    parser.add_argument('-e', '--test', action='store_true', help='rename testing files')
    parser.add_argument('-r', '--train', action='store_true', help='rename trained-data files')
    parser.add_argument('-p', '--path', metavar='', type=str, help='path to the trained-data data from the model directory')
    args = parser.parse_args()

    rename(args)

