# Core modules
import shutil
import tempfile
from typing import List

# Project modules
from .video_utility import sync_video, whash_video
from .scene_parser import extract_scenes, compare_scenes, sync_scenes


def video_similarity(reference_path:str, comparative_path:str, threshold=6.0) -> float:
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


def video_merge(reference_path: str, comparative_path: str, dest: str, threshold=6.0):
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


def video_hash(filename:str) -> List[str]:
    # Hash video
    result = whash_video(filename)

    # Fetch only the hash and not the timestamp
    return [item[1] for item in result]
