# Core modules
import os

# Project modules
from src.__init__ import video_similarity, video_merge
from src.scene_parser import find_optimal_threshold

# Constants
ORIGINAL_FILE = os.path.join('test', 'B', 'original.mp4')
REFERENCE_FILE = os.path.join('test', 'B', 'cut1.mp4')
COMPARE_FILE = os.path.join('test', 'B', 'cut2.mp4')
MERGED_FILE = os.path.join('test', 'B', 'result.mp4')

# Find optimal threshold for reference file
threshold = find_optimal_threshold(REFERENCE_FILE)
print(f'Optimal threshold is: {threshold:.2f}')

# Find similarity between videos
similarity = video_similarity(REFERENCE_FILE, COMPARE_FILE, threshold)
print(f'Video similarity: {similarity * 100:.2f}%')

# Merge videos
if similarity > 0:
    video_merge(REFERENCE_FILE, COMPARE_FILE, MERGED_FILE, threshold)
    print('Video merged')

# Check similarity with original file
if os.path.exists(ORIGINAL_FILE):
    similarity = video_similarity(ORIGINAL_FILE, MERGED_FILE, threshold)
    print(f'Video similarity from original: {similarity * 100:.2f}%')
