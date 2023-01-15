"""
Choose files from a directory
"""

import os
import random


def choose_file(directory):
    """
    Choose a random file from a directory
    """
    files = os.listdir(directory)
    return random.choice(files)


def choose_files(directory, number, deep=False, filter=None, recursive=False):
    """
    Choose a random file from a directory
    """
    if deep:
        files = []
        for file in os.listdir(directory):
            if is_directory(os.path.join(directory, file)):
                files += choose_files(os.path.join(directory, file),
                                      number, deep, filter, recursive)
            elif filter(file):
                files.append(file)
        if recursive:
            return files
        else:
            return random.sample(files, min(number, len(files)))
    else:
        files = os.listdir(directory)
        if filter:
            files = [file for file in files if filter(file)]
        return random.sample(files, min(number, len(files)))


def is_image(file):
    """
    Check if a file is an image
    """
    return file.endswith('.jpg') or file.endswith('.png')


def is_directory(file):
    """
    Check if a file is a directory
    """
    return os.path.isdir(file)


if __name__ == '__main__':
    print(choose_file('.'))
    print(is_directory('.'))
    print(choose_files('data', 10, deep=True, filter=is_image))
