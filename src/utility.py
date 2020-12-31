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


def np_whash(array:np.array, scale=1.5) -> str:
    """Calculate the wavelenghted perceptual hash of a numpy array (image)

    Args:
        array (np.array): Numpy array to hash
        scale (float): Scale reduction of the image. Default to 1.5

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


def compare_videohash(ref_hash_list: List[FrameHash], cmp_hash_list: List[FrameHash], threshold=0.1) -> bool:
    """Compare the hashes of two videos to determine if they are similar

    Args:
        ref_hash_list (List[FrameHash]): Reference video hash list
        cmp_hash_list (List[FrameHash]): Comparison video hash list
        threshold (float, optional): Value above which to consider two hash lists 
                                    (and consequently two videos) similar. 
                                    Defaults to 0.1.

    Returns:
        bool: True if the two videos are similar, False otherwise
    """
    
    # Local variables
    count = 0

    for a in cmp_hash_list:
        (_, cmp_hash) = a
        results = [item for item in ref_hash_list if cmp_hash in item]
        if len(results) > 0:
            count += 1

    similarity = count / len(ref_hash_list)
    similarity = min(similarity, 1.0)

    return similarity > threshold
