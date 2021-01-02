# Core modules
import os

# Project modules
from src.__init__ import video_similarity, video_merge

# Constants
BASE_PATH = os.path.join('test', 'C')
ORIGINAL_FILE = os.path.join(BASE_PATH, 'original.mp4')
REFERENCE_FILE = os.path.join(BASE_PATH, 'cut1.mp4')
COMPARE_FILE = os.path.join(BASE_PATH, 'cut2.mp4')
MERGED_FILE = os.path.join(BASE_PATH, 'result.mp4')

# Find similarity between videos
similarity = video_similarity(REFERENCE_FILE, COMPARE_FILE)
print(f'Video similarity: {similarity * 100:.2f}%')

# Merge videos
if similarity > 0:
    video_merge(REFERENCE_FILE, COMPARE_FILE, MERGED_FILE)
    print('Video merged')
    
# Check similarity with original file
if os.path.exists(ORIGINAL_FILE):
    similarity = video_similarity(ORIGINAL_FILE, MERGED_FILE)
    print(f'Video similarity from original: {similarity * 100:.2f}%')
