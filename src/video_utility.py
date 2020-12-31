# Core modules
from typing import Tuple, List

# pip modules
from moviepy.editor import VideoFileClip, concatenate_videoclips

# Project modules
from .utility import np_whash, np_md5, compare_videohash

# Aliases and types
FrameHash = Tuple[float, str]
TimeFrame = Tuple[float, List]


def _get_frame_list(filepath:str, n=5) -> List[TimeFrame]:
    """Gets frames from a video file

    Args:
        filepath (str): Path to video
        n (int, optional): Number of frames each to process a frame. Defaults to 5.

    Returns:
        List[TimeFrame]: List of tuples in the form (timestamp, frame)
    """
    
    # Local variables
    data = []
    t = 0
    
    with VideoFileClip(filepath) as clip:
        # Find the time interval after which to get a frame
        time_slice = clip.duration / n
        
        while t < clip.duration:
            # Get frame at current time
            frame = clip.get_frame(t)
            
            # Save FrameHash
            data.append((t, frame))
            
            # Increment time by time_slice
            t += time_slice
    
    return data


def _find_sync_point(reference_data:List[FrameHash], 
                    compare_data:List[FrameHash], 
                    matching_frames=15) -> Tuple[FrameHash, FrameHash]:
    """Find the point to join two videos with the same frames

    Args:
        reference_data (list[FrameHash]): Frame hash list of the reference video
        compare_data (list[FrameHash]): List of hashes of frames of the compared video
        matching_frames (int, optional): Number of frames that must be 
                                        consecutively equal in order to consider 
                                        the synchronization point. Defaults to 15.

    Returns:
        tuple[FrameHash, FrameHash]: Synchronization point, the first element refers 
                                    to the reference video while the second to the compared video.
                                    If the point is not found, it returns None
    """
    # Local variables
    sync_point = None
    sync_count = 0

    for i in reference_data:
        (ref_ts, ref_hash) = i

        # Check if the number of sync frames are enough
        if sync_count == matching_frames: break

        for cmp in compare_data:
            (ts, hash) = cmp

            # Check if the number of sync frames are enough
            if sync_count == matching_frames: break

            if ref_hash == hash:
                # Assig the sync point (if not exists)
                if sync_point is None:
                    sync_point = (i, cmp)

                # Increment counter
                sync_count += 1
            else:
                # Frames mismatch, this is not a sync point
                # reset the values
                sync_point = None
                sync_count = 0
    
    return sync_point


def whash_video(filepath:str, frame_skip=5) -> List[FrameHash]:
    """Gets the wavelenghted perceptual hashes of the frames (and their timestamps) of a video.

    Args:
        filepath (str): Path to video
        frame_skip (int, optional): Every how many frames calculate a hash. Defaults to 5.

    Returns:
        list[FrameHash]: List of tuples (timestamp, hash frame) of type (float, str)
    """

    # Local variables
    data_list = []
    tuple_list = []
    
    # Obtains both the frame and the timestamp of it
    for data in _get_frame_list(filepath, frame_skip):
        (timestamp, frame) = data
        hash = np_whash(frame)
        tuple_list.append((timestamp, hash))

    return tuple_list


def compare_video(reference_path:str, video_path:str, frame_skip=5) -> float:
    """Calculate the similarity between two videos 
        by calculating the perceptual hashes each frame_skip frame

    Args:
        reference_path (str): Path to the reference video
        video_path (str): Path to the video to compare
        frame_skip (int, optional): Every how many frames calculate a hash. Defaults to 5.

    Returns:
        float: Similairty of the videos (from 0.0 to 1.0)
    """
    
    # Obtains the hash of the frames in the videos
    reference_hash_list = whash_video(reference_path, frame_skip)
    video_hash_list = whash_video(video_path, frame_skip)

    # Compare hashes
    return compare_video_hash(reference_hash_list, video_hash_list)


def compare_video_hash(reference_hash_list: List[FrameHash], compare_hash_list: List[FrameHash]) -> float:
    """It compares the perceptual hashes of two videos 
        and returns a similarity index between 0 and 1

    Args:
        reference_hash_list (List[FrameHash]): Hash of the reference video
        compare_hash_list (List[FrameHash]): Hash of the video to compare

    Returns:
        float: Video similarity index. Between 0 and 1
    """
    
    # Local variables
    count = 0
    
    # Search for video hash in the reference_hash_list
    for data in compare_hash_list:
        (timestamp, hash) = data
        results = [item for item in reference_hash_list if hash in item]
        count += len(results)

    return count / len(reference_hash_list)


def sync_video(reference_path:str, compare_path:str, dest:str) -> bool:
    """Concatenates two videos belonging to the same movie based on equal frames

    Args:
        reference_path (str): Reference video path
        compare_path (str): Compared video path
        dest (str): Result video path

    Returns:
        bool: Result of the operation
    """
    # Hash videos (perceptual)
    ref_hash_list = whash_video(reference_path, frame_skip=1)
    cmp_hash_list = whash_video(compare_path, frame_skip=1)

    # Find sync point
    sync_point = _find_sync_point(ref_hash_list, cmp_hash_list)
    if sync_point is None: return False

    # Read video clips
    ref_clip = VideoFileClip(reference_path)
    cmp_clip = VideoFileClip(compare_path)

    # Slice clips based on sync point
    (ref_point, cmp_point) = sync_point
    start_clip = ref_clip.subclip(0, ref_point[0])
    end_clip = cmp_clip.subclip(cmp_point[0])

    # Merge and save clip
    final = concatenate_videoclips([start_clip, end_clip], method='compose')
    final.write_videofile(dest, preset='fast', threads=4, verbose=False, logger=None)
    final.close()

    # Close all the clips
    for clip in [start_clip, end_clip, ref_clip, cmp_clip]:
        clip.close()

    return True
