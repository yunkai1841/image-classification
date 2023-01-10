"""
Image resize tools
"""
import os
import glob
from PIL import Image
from torchvision import transforms

def to_tensor(image):
    """
    Convert an image to a tensor.

    Parameters
    ----------
    image : PIL.Image
        Image to convert.

    Returns
    -------
    torch.Tensor
        Image as a tensor.
    """
    return transforms.ToTensor()(image)

def add_margin(image, color, size):
    """
    Add a margin to an image.

    Parameters
    ----------
    image : PIL.Image
        Image to add a margin to.
    color : tuple
        Color of the margin.
    size : tuple
        Size of the margin.

    Returns
    -------
    PIL.Image
        Image with a margin.
    """
    width, height = image.size
    new_width = width + 2 * size[0]
    new_height = height + 2 * size[1]
    new_image = Image.new('RGB', (new_width, new_height), color)
    new_image.paste(image, size)

    return new_image

def to_square(image, color=(0, 0, 0)):
    """
    Resize an image to a square.

    Parameters
    ----------
    image : PIL.Image
        Image to resize.
    color : tuple
        Color of the margin.

    Returns
    -------
    PIL.Image
        Resized image.
    """
    width, height = image.size
    #! TODO: fix even-odd bug (e.g. 256x255)
    if width > height:
        image = add_margin(image, color, (0, (width - height) // 2))
    elif height > width:
        image = add_margin(image, color, ((height - width) // 2, 0))

    return image

def crop_image(image, size):
    """
    Crop an image to a given size.

    Parameters
    ----------
    image : PIL.Image
        Image to crop.
    size : tuple
        Size of the output image.

    Returns
    -------
    PIL.Image
        Cropped image.
    """
    width, height = image.size
    left = (width - size[0]) // 2
    top = (height - size[1]) // 2
    right = (width + size[0]) // 2
    bottom = (height + size[1]) // 2

    return image.crop((left, top, right, bottom))

def resize_image(image, size, keep_aspect_ratio=True):
    """
    Resize an image to a given size.

    Parameters
    ----------
    image : PIL.Image
        Image to resize.
    size : tuple
        Size of the output image.
    keep_aspect_ratio : bool
        If True, keep the aspect ratio of the image.

    Returns
    -------
    PIL.Image
        Resized image.
    """
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
    keep_aspect_ratio: bool = True,
    postprocess: callable = None):
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
    postprocess : callable
        Function to apply to the image after resizing.
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
                    if postprocess is not None:
                        image = postprocess(image)
                    image.save(os.path.join(output_dir, file))
    else:
        for file in glob.glob(os.path.join(input_dir, '*')):
            if file.endswith(exts):
                image = Image.open(file)
                image = resize_image(image, size, keep_aspect_ratio)
                if postprocess is not None:
                    image = postprocess(image)
                image.save(os.path.join(output_dir, os.path.basename(file)))


if __name__ == '__main__':
    resize_images('images', 'resized_images', (256, 256), postprocess=to_square)