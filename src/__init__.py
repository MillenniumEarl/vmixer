# Core modules
import shutil
import tempfile
from typing import List, Tuple

# Project modules
from .utility import videohash_similarity
from .video_utility import sync_video, whash_video
from .scene_parser import extract_scenes, compare_scenes, sync_scenes, cache_dir

# Aliases and types
FrameHash = Tuple[float, str]


def video_similarity(reference_path: str, comparative_path: str, threshold=10.0) -> float:
    """Compare two videos to find similarities. Returns a value 
        between 0 and 1, where 0 indicates no similarity and 1 returns equality

        The first video passed by parameter is taken as a reference, 
        inverting the videos could change the result

    Args:
        reference_path (str): Path to reference video
        comparative_path (str): Path of the video to compare
        threshold (float, optional): Threshold used to detect scene change. 
                                    The smaller it is, the more scenes are detected. 
                                    Defaults to 10.0.

    Returns:
        float: Similarity of videos, value between 0 and 1
    """

    # Local variables
    tmp_ref_dest = tempfile.mkdtemp()
    tmp_cmp_dest = tempfile.mkdtemp()

    # Split videos in scenes
    ref_scene_data = extract_scenes(reference_path, tmp_ref_dest, threshold)
    cmp_scene_data = extract_scenes(comparative_path, tmp_cmp_dest, threshold)

    # Compare scenes
    (similarity, pairs) = compare_scenes(ref_scene_data, cmp_scene_data)

    # Delete temp scenes
    shutil.rmtree(tmp_ref_dest)
    shutil.rmtree(tmp_cmp_dest)

    return similarity


def video_hash_similarity(reference_hash_list: List[FrameHash], compare_hash_list: List[FrameHash]) -> float:
    """It compares the perceptual hashes of two videos 
        and returns a similarity index between 0 and 1

    Args:
        reference_hash_list (List[FrameHash]): Hash of the reference video
        compare_hash_list (List[FrameHash]): Hash of the video to compare

    Returns:
        float: Video similarity index. Between 0 and 1
    """
    return videohash_similarity(reference_hash_list, compare_hash_list)


def video_merge(reference_path: str, comparative_path: str, dest: str, threshold=6.0):
    """Compare two videos and synchronize them on common frames, 
        creating a single synchronized video

    Args:
        reference_path (str): Path to reference video
        comparative_path (str): Path of the video to compare
        dest (str): Destination of the merged video
        threshold (float, optional): Threshold used to detect scene change. 
                                    The smaller it is, the more scenes are detected. 
                                    Defaults to 6.0.
    """
    # Local variables
    tmp_ref_dest = tempfile.mkdtemp()
    tmp_cmp_dest = tempfile.mkdtemp()

    # Split videos in scenes
    ref_scene_data = extract_scenes(reference_path, tmp_ref_dest, threshold)
    cmp_scene_data = extract_scenes(comparative_path, tmp_cmp_dest, threshold)

    # Merge videos
    sync_scenes(ref_scene_data, cmp_scene_data, dest)

    # Delete temp scenes
    shutil.rmtree(tmp_ref_dest)
    shutil.rmtree(tmp_cmp_dest)


def video_hash(filename: str) -> List[str]:
    """Calculate the perceptual hashes of a video

    Args:
        filename (str): Path of the video

    Returns:
        List[str]: List of hashes that identify the video
    """
    # Hash video
    result = whash_video(filename)

    # Fetch only the hash and not the timestamp
    return [item[1] for item in result]
