import os
import re


def SwapFiles(files, source, destination):
    """

    :param files:
    :param source:
    :param destination:
    :return:
    """
    files[source], files[destination] = files[destination], files[source]
    return files


def partition(array, low, high, files):
    left = (low - 1)  # index of smaller element
    pivot = array[high]  # pivot

    for right in range(low, high):

        # If current element is smaller than or
        # equal to pivot
        if array[right] <= pivot:
            # increment index of smaller element
            left = left + 1
            array[left], array[right] = array[right], array[left]
            if files:
                SwapFiles(files, left, right)

    array[left + 1], array[high] = array[high], array[left + 1]
    if files:
        SwapFiles(files, left + 1, high)
    return left + 1


def QuickSort(array, low, high, files=None):
    if len(array) == 1:
        return array
    if low < high:
        # pi is partitioning index, arr[p] is now
        # at right place
        pi = partition(array, low, high, files=files)

        # Separately sort elements before
        # partition and after partition
        QuickSort(array, low, pi - 1, files=files)
        QuickSort(array, pi + 1, high, files=files)


def CreateMappings(files, sep):
    """
    map the files label with index in the array
    :param files: list of files
    :param sep: separator used in the files e.g. '-'
    :return:
    """
    mapping = []
    print('creating mappings')
    for file in files:
        name, ext = os.path.splitext(file)
        index = FileIndex(name, sep)

        mapping.append(index)
    return mapping


def sortFiles(files, sep='-'):
    mapping = CreateMappings(files, sep=sep)
    length = len(mapping)

    print(files)

    # sorting files
    QuickSort(mapping, 0, length - 1, files=files)
    return files


def FileIndex(file_name, sep):
    elements = file_name.split(sep)
    return elements


def getFiles(path, pattern):
    files = os.listdir(path)

    Unsorted, Sorted = [], []

    regex = re.compile(pattern)
    for index, file in enumerate(files.copy()):
        file_name, file_ext = os.path.splitext(file)
        match = regex.match(file_name)
        if match:
            temp = files[index]
            Sorted.append(temp)
        else:
            temp = files[index]
            Unsorted.append(temp)

    return Unsorted, Sorted


if __name__ == '__main__':
    arr = [10, 7, 8, 9, 1, 5]
    n = len(arr)
    QuickSort(arr, 0, n - 1)
    print("Sorted array is:")
    for i in range(n):
        print("%d" % arr[i])
