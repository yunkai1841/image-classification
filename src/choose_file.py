"""
Choose files from a directory
"""

import os
import random
import shutil


def choose_file(directory):
    """
    Choose a random file from a directory
    """
    files = os.listdir(directory)
    return random.choice(files)


def choose_files(directory, number, deep=False, filter=(lambda x: True), recursive=False):
    """
    Choose a random file from a directory
    """
    if deep:
        files = []
        for file in os.listdir(directory):
            if is_directory(os.path.join(directory, file)):
                files += choose_files(os.path.join(directory, file),
                                      number, deep, filter, recursive=True)
            elif filter(file):
                files.append(os.path.join(directory, file))
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


def move_files(files, source, destination):
    """
    Move files from a directory to another
    """
    for file in files:
        shutil.move(os.path.join(source, file), destination)

if __name__ == '__main__':
    files = choose_files('resized_images/danbooru', 10000, deep=True)
    move_files(files, '', 'data/danbooru')