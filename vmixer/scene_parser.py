# Core modules
import os
import shutil
import glob
import tempfile
from typing import Tuple, List
import uuid
import csv

# pip modules
from scenedetect import VideoManager
from scenedetect import SceneManager
from scenedetect import StatsManager
from scenedetect import video_splitter
from scenedetect.detectors import ContentDetector
from moviepy.editor import VideoFileClip, concatenate_videoclips
from joblib import Parallel, delayed

# Project imports
from .video_utility import phash_video, compare_video, sync_video
from .utility import md5, compare_videohash, videohash_similarity

# Aliases and types
FrameHash = Tuple[float, str]
SceneData = Tuple[str, int, List[FrameHash]]
SceneHash = Tuple[str, int, List[FrameHash]]
ComparationResult = Tuple[float, List[Tuple[str, str]]]
ScenePairPath = Tuple[str, str]

# Create a folder for the video stats
cache_dir = os.path.join(os.getenv('APPDATA'), 'vmixer_cache')
if not os.path.exists(cache_dir):
    os.mkdir(cache_dir)


def _chunks(lst, n):
    """Yield successive n-sized chunks from lst."""
    for i in range(0, len(lst), n):
        yield lst[i:i + n]


def _merge_chunk_clips(paths: List[str], preset='ultrafast') -> str:
    """Merges all the videos identified by the paths contained in the parameter
        in a single file

    Args:
        paths (List[str]): List of video paths
        preset (str): Preset to use for concatenating the videos

    Returns:
        str: Path to merged video
    """
    # Read the clip
    merge_clips = [VideoFileClip(path) for path in paths]

    # Merge the clips
    tmp_dest = os.path.join(tempfile.mkdtemp(), f'{str(uuid.uuid4().hex)}.mp4')
    tmp_clip = concatenate_videoclips(merge_clips, method='compose')

    # Save the clip to disk
    tmp_clip.write_videofile(tmp_dest, threads=4, preset=preset,
                             ffmpeg_params=['-crf', '17'],
                             verbose=False, logger=None)

    # Close the clips
    tmp_clip.close()
    map(lambda clip: clip.close(), merge_clips)

    return tmp_dest


def _hash_scene(filename: str) -> SceneHash:
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
    hash = phash_video(filename)

    return (filename, index, hash)


def _scene_sync_point(reference_data: List[SceneData], compare_data: List[SceneData]):

    # Local variables
    similar_scenes = []

    for ref_data in reference_data:
        (ref_path, _, ref_hash) = ref_data

        for cmp_data in compare_data:
            (cmp_path, _, cmp_hash) = cmp_data

            # Get the similarity index between video files
            similarity = videohash_similarity(ref_hash, cmp_hash)

            if similarity > 0:
                # Append the similarity point found
                point = (ref_path, cmp_path)
                similar_scenes.append((similarity, point))

    # Sort for better matching
    similar_scenes.sort(
        key=lambda element: element[0], reverse=True)  # Sorts in place

    return similar_scenes[0][1] if len(similar_scenes) > 0 else None


def _create_scenes_map_old(reference_data: List[SceneData], compare_data: List[SceneData]) -> List[ScenePairPath]:
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
    first_duplicate = False
    comparative_before_reference = False
    cmp_before_ref_index = 0

    # Join the list in a single consecutive list
    timeline = reference_data + compare_data

    # Starting from the first element:
    # If no duplicate scene is found: add path to scene_map
    # If duplicate scene is found: add both paths to scene_map
    #
    # When the first duplicate is found, if the duplicate's index is
    # NOT equal to len(reference_data) + 1 (so if the duplicate scene
    # is at the END of compare_data) set comparative_before_reference
    # to True and add the data of **compare_data** to scene_map from
    # the start of the list (index 0)

    for i in range(0, len(timeline)):
        # In-loop variables
        duplicate_found = False

        # Unpack reference data
        (path, _, hash) = timeline[i]

        # Check if this scene is already in the scene_map
        # (only after the join point, before is pointless as
        # there aren't (hopefully) duplicate scenes in the same video)
        if i >= len(reference_data):
            if sum(1 for item in scene_map if path in item):
                continue

        # Search for the first duplicate (only if we are in
        # the data belonging to reference_data)
        if i < len(reference_data):
            for cmpi in range(len(reference_data), len(timeline)):
                # Unpack comparative data
                (cmp_path, _,  cmp_hash) = timeline[cmpi]

                # Check if this scene is already in the scene_map
                if sum(1 for item in scene_map if cmp_path in item):
                    continue

                # Check similarity between scenes
                if compare_videohash(hash, cmp_hash, threshold=0.1):
                    print(
                        f'{path} -> {cmp_path} : {videohash_similarity(hash, cmp_hash)}')
                    duplicate_found = True

                    # Append pair to scene_map
                    scene_map.append((path, cmp_path))

                    # Manage first duplicate
                    if not first_duplicate:
                        first_duplicate = True

                        # Check if the comparative data starts
                        # **before** the reference data
                        index_duplicate = compare_data.index(timeline[cmpi])
                        ten_percent = int(len(compare_data) / 10)
                        comparative_before_reference = index_duplicate > ten_percent

                    # Exit from loop
                    break

        # Add element at index i if no duplicate was found
        if not duplicate_found:
            item = (path, None)

            # Add the item at the beginning of the list,
            # mantaining the time order. Valid only for the
            # compare_data elements
            if comparative_before_reference and i >= len(reference_data):
                scene_map.insert(cmp_before_ref_index, item)
                cmp_before_ref_index += 1

            # Add item at the end of the list
            else:
                scene_map.append(item)

    return scene_map


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
    sync_value = _scene_sync_point(reference_data, compare_data)

    if sync_value is None:
        # Simply merge both videos
        merged = reference_data + compare_data
        scene_map = [(item[0], None) for item in merged]
    else:
        # Obtains the indexes
        (ref_path, cmp_path) = sync_value
        ref_index = reference_data.index(
            [item for item in reference_data if ref_path in item][0])
        cmp_index = compare_data.index(
            [item for item in compare_data if cmp_path in item][0])

        # Get the elements NOT synced
        first_part = [(item[0], None) for item in reference_data[:ref_index]]
        second_part = [(item[0], None)
                       for item in compare_data[cmp_index + 1:]]

        scene_map = first_part + [sync_value] + second_part

    return scene_map


def find_optimal_threshold(filename, frames_per_scene=30):
    # Local variables
    BASE_THRESHOLD = 5.0
    content_val_list = []

    # Load cache
    filehash = md5(filename)
    cache_file = os.path.join(cache_dir, f'{filehash}.csv')

    # If the cache doesn't exists, it will be created
    if not os.path.exists(cache_file):
        find_scenes(filename)

    # Read the CSV cache
    with open(cache_file, 'r') as f:
        #framerate = float(f.readline().rstrip('\n').split(',')[1])
        f.readline()  # Skip framerate
        csv_file = csv.DictReader(f)
        content_val_list = [float(row['content_val']) for row in csv_file]

    # Get the number of scenes available with the given parameters
    desired_scenes = int(len(content_val_list) / frames_per_scene)

    # Define the base values for the binary search
    max_threshold = max(content_val_list)
    base_threshold = 0
    optimal_threshold = max(content_val_list)
    last_delta = optimal_threshold

    # Exit from the loop when we have reached a stale point
    while last_delta > 0.01:
        # Count the number of scenes available with this threshold
        available_scenes = sum(
            val >= optimal_threshold for val in content_val_list)

        # Tune the threshold
        if available_scenes < desired_scenes:
            max_threshold = optimal_threshold
        elif available_scenes > desired_scenes:
            base_threshold = optimal_threshold
        else:
            break

        # Save the current threshold value
        prec_value = optimal_threshold

        # Calcolate the optimal threshold
        optimal_threshold = (max_threshold + base_threshold) / 2

        # Calcolate the delta
        last_delta = abs(optimal_threshold - prec_value)

    return max(optimal_threshold, BASE_THRESHOLD)


def find_scenes(video_path: str, threshold=30.0, cache=True) -> List:
    """Find all the scenes in a video.

    Args:
        video_path (str): Path to the video
        threshold (float, optional): Threshold used to detect scene change. Defaults to 30.0.

    Returns:
        list: List of tuple, each one rapresenting a single scene
    """

    # Local variables
    filehash = None

    # Create our video & scene managers, then add the detector.
    video_manager = VideoManager([video_path])
    stats_manager = StatsManager()
    scene_manager = SceneManager(stats_manager)
    scene_manager.add_detector(ContentDetector(threshold=threshold))

    # Base timestamp at frame 0 (required to obtain the scene list).
    base_timecode = video_manager.get_base_timecode()

    # Check if the file is in the cache
    if cache:
        filehash = md5(video_path)
        cache_file = os.path.join(cache_dir, f'{filehash}.csv')

        if os.path.exists(cache_file):
            with open(cache_file, 'r') as f:
                stats_manager.load_from_csv(f, base_timecode=base_timecode)

    # Improve processing speed by downscaling before processing.
    video_manager.set_downscale_factor()

    # Start the video manager and perform the scene detection.
    video_manager.start()
    scene_manager.detect_scenes(
        frame_source=video_manager, show_progress=False)

    # Save stats of the file
    if stats_manager.is_save_required() and cache:
        cache_file = os.path.join(cache_dir, f'{filehash}.csv')
        with open(cache_file, 'w') as f:
            stats_manager.save_to_csv(f, base_timecode)

    # Each returned scene is a tuple of the (start, end) timecode.
    return scene_manager.get_scene_list(base_timecode)


def extract_scenes(video_path: str, output_dir: str, threshold=10.0) -> List[SceneData]:
    """Extract from a video file all the scenes and save them in the output directory.

    Args:
        video_path (str): Path to the video
        output_dir (str): Directory where to save the extracted scenes
        threshold (float, optional): Threshold used to detect scene change. Defaults to 30.0.

    Returns:
        list[SceneData]: List of tuple in the format (scene path, scene index, perceptual video hashes)
    """

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
            '',  # Video name is blank because is not needed
            # Changed the preset and the CRF (21 -> 17)
            arg_override='-c:v libx264 -preset ultrafast -crf 17 -c:a aac',
            hide_progress=True,
            suppress_output=True)

    # Calculate hash list
    pattern = os.path.join(output_dir, '*.mp4')
    hash_list = Parallel(n_jobs=-1, prefer='threads')(delayed(_hash_scene)(path)
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
        (path, _, hash) = data

        # Get only the scenes similar to data with a minimum similarity,
        # then sort the list and get the most similar element
        similarity_map = [(videohash_similarity(item[2], hash), item)
                          for item in reference_data]
        similarity_map = [item for item in similarity_map if item[0] > 0]

        # Sort to get the most similar scene
        similarity_map.sort(key=lambda element: element[0])

        # Get the first result
        similar = similarity_map[0] if len(similarity_map) > 0 else None

        # Calculate similarity
        if similar is not None:
            # Unpack data
            (similarity, data) = similar
            (ref_path, _, _) = data

            count += similarity
            pairs.append((ref_path, path))

    # Calculate similarity (how many files are similar)
    similarity = count / len(compare_data)
    similarity = min(similarity, 1.0)

    return (similarity, pairs)


def sync_scenes(reference_data: List[SceneData], compare_data: List[SceneData], dest: str):
    """Synchronize scenes belonging to two videos into a single video

    Args:
        reference_data (List[SceneData]): List of scene data of the reference video
        compare_data (List[SceneData]): List of data related to the scenes of the comparison video
        dest (str): Merged video path
    """

    # Local variables
    MAX_CLIP_AT_TIME = 10
    scene_map = _create_scenes_map(reference_data, compare_data)
    clip_paths = []
    temp_video = []  # Path of videos to be deleted

    for pair in scene_map:
        # Unpack the paths
        (path, merge_path) = pair

        if merge_path is None:
            clip_paths.append(path)
        else:
            tmp_dir = tempfile.mkdtemp()
            tmp_dest = os.path.join(tmp_dir, f'{str(uuid.uuid4().hex)}.mp4')
            temp_video.append(tmp_dir)
            if sync_video(path, merge_path, tmp_dest):
                clip_paths.append(tmp_dest)

    # Merge chunks of clips until MAX_CLIP_AT_TIME remains
    while len(clip_paths) > MAX_CLIP_AT_TIME:
        chunk_clips_paths = Parallel(n_jobs=-1, prefer='processes')(delayed(_merge_chunk_clips)(chunk)
                                                                    for chunk in _chunks(clip_paths, MAX_CLIP_AT_TIME))

        # Add paths to list of deletable videos
        temp_video.extend(chunk_clips_paths)

        # Set clips path
        clip_paths = chunk_clips_paths

    # Merge all clips in a single video
    tmp_final_path = _merge_chunk_clips(clip_paths, preset='medium')

    # Move the file to dest
    shutil.move(tmp_final_path, dest)

    # Delete all the merged video directories
    temp_video.extend(clip_paths)
    dirs = [os.path.dirname(file) for file in temp_video]
    map(shutil.rmtree, dirs)
