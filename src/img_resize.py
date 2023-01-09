"""
Resize images to a given size.
"""
import os
import glob
from PIL import Image


def resize_image(image, size, keep_aspect_ratio=True):
    if keep_aspect_ratio:
        image.thumbnail(size, Image.Resampling.LANCZOS)
    else:
        image = image.resize(size, Image.Resampling.LANCZOS)

    return image


def resize_images(
    input_dir: str,
    output_dir: str,
    size: tuple,
    deep: bool = False,
    exts: tuple = ('.jpg', '.png'),
    keep_aspect_ratio: bool = True):
    """
    Resize images in a given directory to a given size.

    Parameters
    ----------
    input_dir : str
        Path to the input directory.
    output_dir : str
        Path to the output directory.
    size : tuple
        Size of the output image.
    deep : bool
        If True, recursively search for images in subdirectories.
    exts : tuple
        Tuple of file extensions to search for.
    keep_aspect_ratio : bool
        If True, keep the aspect ratio of the image.
    """
    if not os.path.exists(input_dir):
        raise Exception('Input directory does not exist')

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    if deep:
        for root, dirs, files in os.walk(input_dir):
            for file in files:
                if file.endswith(exts):
                    image = Image.open(os.path.join(root, file))
                    image = resize_image(image, size, keep_aspect_ratio)
                    image.save(os.path.join(output_dir, file))
    else:
        for file in glob.glob(os.path.join(input_dir, '*')):
            if file.endswith(exts):
                image = Image.open(file)
                image = resize_image(image, size, keep_aspect_ratio)
                image.save(os.path.join(output_dir, os.path.basename(file)))


if __name__ == '__main__':
    resize_images('images', 'resized_images', (256, 256))