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
from joblib import Parallel, delayed

# Project imports
from .video_utility import whash_video, compare_video, sync_video
from .utility import compare_videohash

# Aliases and types
FrameHash = Tuple[float, str]
SceneData = Tuple[str, int, List[FrameHash]]
SceneHash = Tuple[str, int, List[FrameHash]]
ComparationResult = Tuple[float, List[Tuple[str, str]]]
ScenePairPath = Tuple[str, str]


def _hash_scene(filename:str) -> SceneHash:
    """Hash a scene (video) and return its data

    Args:
        filename (str): Path to the video

    Returns:
        SceneHash: Tuple in the format(scene path, scene index, scene hash)
    """
    
    # Get the index of the scene in the video
    scene_name = os.path.splitext(os.path.basename(filename))[0]
    index = int(scene_name)
    
    # Hash the scene
    hash = whash_video(filename)
    
    return (filename, index, hash)


def _create_scenes_map(reference_data: List[SceneData], compare_data: List[SceneData]) -> List[ScenePairPath]:
    """Create the sorted list of scenes to merge to get the final video

    Args:
        reference_data (List[SceneData]): List of scene data of the reference video
        compare_data (List[SceneData]): List of data related to the scenes of the comparison video

    Returns:
        List[ScenePairPath]: Ordered list of tuples, each containing the video path 
                            to include in the final video. If the tuple contains two paths, 
                            the related videos must be merged
    """
    
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
    scene_manager.detect_scenes(frame_source=video_manager, show_progress=True)

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
            hide_progress=False,
            suppress_output=True)

    # Calculate hash list
    pattern = os.path.join(output_dir, '*.mp4')
    hash_list = Parallel(n_jobs=-1, backend='threading')(delayed(_hash_scene)(path)
                                                         for path in glob.glob(pattern))
    return hash_list


def compare_scenes(reference_data: List[SceneData], compare_data: List[SceneData]) -> ComparationResult:
    """Compare the scenes that make up a video by returning 
        a similarity index between 0 and 1 and the list 
        of similar scenes

    Args:
        reference_data (List[SceneData]): List of scene data of the reference video
        compare_data (List[SceneData]): List of data related to the scenes of the comparison video

    Returns:
        ComparationResult: Tuple in the format (similarity, list of pairs of similar scenes]
    """
    
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
    similarity = min(similarity, 1.0)

    return (similarity, pairs)


def sync_scenes(reference_data: List[SceneData], compare_data: List[SceneData], dest:str):
    """Synchronize scenes belonging to two videos into a single video

    Args:
        reference_data (List[SceneData]): List of scene data of the reference video
        compare_data (List[SceneData]): List of data related to the scenes of the comparison video
        dest (str): Merged video path
    """
    
    # Local variables
    scene_map = _create_scenes_map(reference_data, compare_data)
    merge_paths = []
    temp_video = []

    for pair in scene_map:
        # Unpack the paths
        (first_path, second_path) = pair

        if first_path is None: merge_paths.append(second_path)
        elif second_path is None: merge_paths.append(first_path)
        else:
            tmp_dir = tempfile.mkdtemp()
            tmp_dest = os.path.join(tmp_dir, 'merged.mp4')
            temp_video.append(tmp_dir)
            if sync_video(first_path, second_path, tmp_dest):
                merge_paths.append(tmp_dest)
        
    # Merge all clips in a single video
    while len(merge_paths) > 1:
        # Load first two clips
        first_clip = VideoFileClip(merge_paths[0])
        second_clip = VideoFileClip(merge_paths[1])
        
        # Create temp directory
        tmp_dir = tempfile.mkdtemp()
        tmp_dest = os.path.join(tmp_dir, 'merged.mp4')
        temp_video.append(tmp_dir)
        
        # Merge two clips
        final = concatenate_videoclips([first_clip, second_clip], method='compose')
        final.write_videofile(tmp_dest, preset='fast', threads=2,
                            verbose=False, logger=None)
        final.close()
        
        # Close clips
        first_clip.close()
        second_clip.close()
        
        # Remove the paths from the list
        merge_paths.remove(merge_paths[0])
        merge_paths.remove(merge_paths[1])
        
        # Add the merged path as first item in the list
        merge_paths.insert(0, tmp_dest)
        
    # Copy the final video
    shutil.move(merge_paths[0], dest)
          
    # Delete all the merged video directories
    for path in temp_video:
        shutil.rmtree(path)
