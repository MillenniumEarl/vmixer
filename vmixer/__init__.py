# Core modules
import shutil
import tempfile
from typing import List, Tuple

# pip modules
from joblib import Parallel, delayed

# Project modules
from .utility import videohash_similarity
from .video_utility import sync_video, whash_video
from .scene_parser import extract_scenes, compare_scenes, sync_scenes, cache_dir, find_optimal_threshold

# Aliases and types
FrameHash = Tuple[float, str]
VideoSimilarity = Tuple[float, str]
SceneData = Tuple[str, int, List[FrameHash]]
ComparisonHash = Tuple[int, float]


def _extract_and_compare_scenes(ref_scene_data: List[SceneData], filename: str, threshold: float) -> VideoSimilarity:
    # Create temp path
    tmp_cmp_dest = tempfile.mkdtemp()

    # Extract comparative scene data
    cmp_scene_data = extract_scenes(filename, tmp_cmp_dest, threshold)

    # Compare scenes
    (similarity, _) = compare_scenes(ref_scene_data, cmp_scene_data)

    # Delete temp scenes
    shutil.rmtree(tmp_cmp_dest)
    
    return (similarity, filename)


def _videohash_similarity_wrapper(reference_hash: List[FrameHash], compare_hash: List[FrameHash], index: int) -> ComparisonHash:
    similarity = videohash_similarity(reference_hash, compare_hash)
    return (index, similarity)


def video_similarity(reference_path:str, *comparative_paths: str, threshold=None) -> List[VideoSimilarity]:
    """Compare multiple videos to find similarities. Returns a list of tuples,
        each containing the path to a comparative video and its similarity 
        with the reference video.

        The first video passed by parameter is taken as a reference, 
        inverting the videos could change the result

    Args:
        reference_path (str): Path to reference video
        *comparative_paths (str): Path(s) of the video to compare
        threshold (float, optional): Threshold used to detect scene change. 
                                    The smaller it is, the more scenes are detected. 
                                    Use None to automatically detect the optimal value.
                                    Defaults to None.

    Returns:
        List[VideoSimilarity]: List of tuple in the format (similarity, path),
                                with similarity index between 0 and 1
    """
    # Get the optimal threshold
    if threshold is None:
        ref_t = find_optimal_threshold(reference_path)
        cmp_ts = [find_optimal_threshold(path) for path in comparative_paths]
        threshold = (sum(cmp_ts) + ref_t) / (len(cmp_ts) + 1)
    
    # Split reference videos in scenes
    tmp_ref_dest = tempfile.mkdtemp()
    ref_scene_data = extract_scenes(reference_path, tmp_ref_dest, threshold)
    
    # Extract data in parallel
    video_similarity = Parallel(n_jobs=-1, backend='threading')(delayed(_extract_and_compare_scenes)(ref_scene_data, path, threshold) for path in comparative_paths)
        
    # Delete reference temp path
    shutil.rmtree(tmp_ref_dest)
        
    return video_similarity


def video_hash_similarity(reference_hash: List[str], *compare_hashes: List[str]) -> List[ComparisonHash]:
    """It compares the perceptual hashes of two or more videos 
        and returns a similarity index between 0 and 1 for each

    Args:
        reference_hash (List[str]): Hash of the reference video
        compare_hashes (List[str]): Hash(es) of the video to compare

    Returns:
        List[ComparisonHash]: Comparison result, a list of tuple in 
                            the format (index of element in compare_hashes, similarity)
    """
    
    # Recreate the lists
    ref_hash_list = [(0, hash) for hash in reference_hash]
    cmp_hash_list = [[(0, hash) for hash in cmp_hash] for cmp_hash in compare_hashes]
    
    # Define the backend used (need to be checked, 100 is random)
    backend = 'threading' if len(cmp_hash_list) <= 100 else None
    
    # Obtains the hashes similarity
    comparison = Parallel(n_jobs=-2, backend=backend)(
        delayed(_videohash_similarity_wrapper)(ref_hash_list, compare_hash, cmp_hash_list.index(compare_hash))
        for compare_hash in cmp_hash_list)
    
    return comparison
    

def video_merge(reference_path: str, comparative_path: str, dest: str, threshold=None):
    """Compare two videos and synchronize them on common frames, 
        creating a single synchronized video

    Args:
        reference_path (str): Path to reference video
        comparative_path (str): Path of the video to compare
        dest (str): Destination of the merged video
        threshold (float, optional): Threshold used to detect scene change. 
                                    The smaller it is, the more scenes are detected. 
                                    Use None to automatically detect the optimal value.
                                    Defaults to None.
    """
    # Local variables
    tmp_ref_dest = tempfile.mkdtemp()
    tmp_cmp_dest = tempfile.mkdtemp()
    
    # Get the optimal threshold
    if threshold is None:
        ref_t = find_optimal_threshold(reference_path)
        cmp_t = find_optimal_threshold(comparative_path)
        threshold = (ref_t + cmp_t) / 2

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