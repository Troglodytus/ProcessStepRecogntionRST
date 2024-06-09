import cv2
import os
import time
from moviepy.video.io.ffmpeg_tools import ffmpeg_extract_subclip

class VideoHandler:

    def __init__(self, video_directory):
        self.video_directory = video_directory

    def load_video(self, video_file):
        video_path = os.path.join(self.video_directory, video_file)
        video = cv2.VideoCapture(video_path)
        return video

    def cut_video(self, video_file, timestamp, duration):
        video_path = os.path.join(self.video_directory, video_file)
        
        # Get the video's creation time
        creation_time = os.path.getmtime(video_path)

        # Calculate the start time for the cut in seconds
        # Assuming creation date is the end of the video
        start_time = creation_time - timestamp

        # Make sure start_time is at least 0
        start_time = max(0, start_time)

        # Calculate the end time for the cut
        end_time = start_time + duration

        # Define the output file name
        output_file = "cut_" + video_file

        # Cut the video using MoviePy's ffmpeg_extract_subclip
        ffmpeg_extract_subclip(video_path, start_time, end_time, targetname=output_file)

        return output_file
