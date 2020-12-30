# Core modules
import os
import shutil
import glob
import tempfile
from typing import Tuple, List

# pip modules
from scenedetect import VideoManager
from scenedetect import SceneManager
from scenedetect import video_splitter
from scenedetect.detectors import ContentDetector
from moviepy.editor import VideoFileClip, concatenate_videoclips

# Project imports
from video_utility import whash_video, compare_video, sync_video
from utility import compare_videohash

# Aliases and types
FrameHash = Tuple[float, str]
SceneData = Tuple[str, int, List[FrameHash]]
ComparationResult = Tuple[float, List[Tuple[str, str]]]
ScenePairPath = Tuple[str, str]


def find_scenes(video_path:str, threshold=30.0) -> List:
    """Find all the scenes in a video.

    Args:
        video_path (str): Path to the video
        threshold (float, optional): Threshold used to detect scene change. Defaults to 30.0.

    Returns:
        list: List of tuple, each one rapresenting a single scene
    """
    # Create our video & scene managers, then add the detector.
    video_manager = VideoManager([video_path])
    scene_manager = SceneManager()
    scene_manager.add_detector(ContentDetector(threshold=threshold))

    # Base timestamp at frame 0 (required to obtain the scene list).
    base_timecode = video_manager.get_base_timecode()

    # Improve processing speed by downscaling before processing.
    video_manager.set_downscale_factor()

    # Start the video manager and perform the scene detection.
    video_manager.start()
    scene_manager.detect_scenes(frame_source=video_manager, show_progress=False)

    # Each returned scene is a tuple of the (start, end) timecode.
    return scene_manager.get_scene_list(base_timecode)


def extract_scenes(video_path: str, output_dir: str, threshold=10.0) -> List[SceneData]:
    """Extract from a video file all the scenes and save them in the output directory.

    Args:
        video_path (str): Path to the video
        output_dir (str): Directory where to save the extracted scenes
        threshold (float, optional): Threshold used to detect scene change. Defaults to 30.0.
    
    Returns:
        list[SceneData]: List of tuple in the format (scene path, scene index, perceptual first frame hash)
    """

    # Local variables
    hash_list = []
    
    # Extract the video name
    filename = os.path.basename(video_path)

    # Create the direcotry where to save the files
    if os.path.exists(output_dir):
        shutil.rmtree(output_dir)
    os.makedirs(output_dir)

    # Extract scenes
    scenes = find_scenes(video_path, threshold)

    # FFmpeg give error with only a single scene,
    # so we copy the original video in the output directory
    if(len(scenes) == 1):
        dst = os.path.join(output_dir, '001.mp4')
        shutil.copyfile(video_path, dst)

    # Split video with FFmpeg
    if(video_splitter.is_ffmpeg_available() and len(scenes) > 1):
        video_splitter.split_video_ffmpeg(
            [video_path],
            scenes,
            os.path.join(output_dir, '$SCENE_NUMBER.mp4'),
            '', # Video name is blank because is not needed
            arg_override='-c:v libx264 -preset ultrafast -crf 21 -c:a aac', # Changed only the preset
            hide_progress=True,
            suppress_output=True)

    # Calculate hash list
    pattern = os.path.join(output_dir, '*.mp4')
    for path in glob.glob(pattern):
        scene_name = os.path.splitext(os.path.basename(path))[0]
        hash = whash_video(path)
        index = int(scene_name)
        hash_list.append((path, index, hash))

    return hash_list


def compare_scenes(reference_data: List[SceneData], compare_data: List[SceneData], threshold=5) -> ComparationResult:
    
    # Local variables
    count = 0
    pairs = []

    for data in compare_data:
        # Unpack data and search the hash in the reference list
        (path, index, hash) = data

        results = [
            item for item in reference_data if compare_videohash(item[2], hash)]

        if len(results) > 0:
            # Find duplicates and ignore them to avoid more match per file
            (ref_path, ref_index, ref_hash) = results[0]
            duplicates = [item for item in pairs if ref_path in item]
            
            # Save the pair of similar files
            if compare_videohash(ref_hash, hash) and len(duplicates) == 0:
                count += compare_video(path, ref_path)
                pairs.append((path, ref_path))

    # Calculate similarity (how many files are similar)
    similarity = count / len(reference_data)

    return (similarity, pairs)


def _create_scenes_map(reference_data: List[SceneData], compare_data: List[SceneData]) -> List[ScenePairPath]:

    # Local variables
    scene_map = []
    cmp_first_index = 0

    # Obtains all the scenes in the reference video 
    # and the common scenes between the videos
    for data in reference_data:
        (path, _, hash) = data
        results = [item for item in compare_data if compare_videohash(hash, item[2])]

        pair = (None, None)
        if len(results) == 0:
            # Scene in the reference video but NOT in the compare video
            pair = (path, None)
        else: 
            # Scene in both the videos, will be merged later
            (cmp_path, _, _) = results[0]
            pair = (path, cmp_path)

            # Save the index that will be later used to 
            # split the compared video data
            index = compare_data.index(results[0])
            if index > cmp_first_index:
                cmp_first_index = index

        scene_map.append(pair)

    # Obtains all the scenes in the comparison video 
    # that are NOT in the reference video
    remainings = compare_data[cmp_first_index + 1:]
    for data in remainings:
        (path, _, _) = data
        scene_map.append((None, path))
    
    return scene_map


def sync_scenes(reference_data: List[SceneData], compare_data: List[SceneData], dest:str):

    # Local variables
    scene_map = _create_scenes_map(reference_data, compare_data)
    merge_clips = []

    for pair in scene_map:
        # In-loop variables
        clip = None

        # Unpack the paths
        (first_path, second_path) = pair

        if first_path is None:
            clip = VideoFileClip(second_path)
        elif second_path is None:
            clip = VideoFileClip(first_path)
        else:
            tmp_dest = os.path.join(tempfile.mkdtemp(), 'merged.mp4')
            sync_video(first_path, second_path, tmp_dest)
            clip = VideoFileClip(tmp_dest)
        
        # Add clip to list of merged clips
        merge_clips.append(clip)
        
    # Merge all clips in a single video
    final = concatenate_videoclips(merge_clips, method='compose')
    final.write_videofile(dest, preset='fast', threads=4,
                          verbose=False, logger=None)
    final.close()

    # Close all the clips
    for clip in merge_clips:
        clip.close()
