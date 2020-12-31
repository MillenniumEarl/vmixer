# Core modules
import os

# Project modules
from src.__init__ import video_similarity, video_merge

# Constants
REFERENCE_FILE = os.path.join('test', 'C', 'cut1.mp4')
COMPARE_FILE = os.path.join('test', 'C', 'cut2.mp4')
MERGED_FILE = os.path.join('test', 'result.mp4')
THRESHOLD = 6.0

# Find similarity between videos
similarity = video_similarity(REFERENCE_FILE, COMPARE_FILE, THRESHOLD)
print(f'Video similarity: {similarity*100:.2f}%')

# Merge videos
if similarity > 0:
    video_merge(REFERENCE_FILE, COMPARE_FILE, MERGED_FILE, THRESHOLD)
    print('Video merged')
