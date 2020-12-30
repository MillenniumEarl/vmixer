# Core modules
import hashlib
from typing import Tuple, List

# pip modules
import imagehash
from PIL import Image
import numpy as np

# Aliases and types
FrameHash = Tuple[float, str]


def md5(filename:str)->str:
    """Calculate the MD5 hash for the specified file

    Args:
        filename (str): Path to the file to hash

    Returns:
        str: MD5 Hash of the file
    """
    hash_md5 = hashlib.md5()

    with open(filename, "rb") as f:
        for chunk in iter(lambda: f.read(4096), b""):
            hash_md5.update(chunk)

    return hash_md5.hexdigest()


def np_whash(array:np.array, scale=2) -> str:
    """Calculate the wavelenghted perceptual hash of a numpy array (image)

    Args:
        array (np.array): Numpy array to hash
        scale (float): Scale reduction of the image. Default to 2

    Returns:
        str: Perceptual hash
    """

    # Convert the image
    image = Image.fromarray(np.uint8(array))

    # Resize the image
    (width, height) = (image.width // scale, image.height // scale)
    image = image.resize((width, height))

    # Calculate hash
    hash = imagehash.whash(image)
    return str(hash)


def np_md5(array) -> str:
    a = array.view(np.uint8)
    return hashlib.md5(a).hexdigest()


def compare_videohash(ref_hash_list: List[FrameHash], cmp_hash_list: List[FrameHash], threshold=0.7):
    # Local variables
    count = 0

    for a in cmp_hash_list:
        (_, cmp_hash) = a
        results = [item for item in ref_hash_list if cmp_hash in item]
        if len(results) >= 1:
            count += 1

    similarity = count / len(ref_hash_list)
    return similarity >= threshold
