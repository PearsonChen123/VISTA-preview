#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Extract Video Frames

This script extracts frames from a video file based on specified start and end times.
Extracted frames are saved as PNG images with the naming format frame_0000.png.

Usage:
    python extract_video_frames.py --video VIDEO_PATH --output OUTPUT_DIR --start START_TIME --end END_TIME

Example:
    python extract_video_frames.py --video input.mp4 --output frames/ --start 00:01:30 --end 00:02:15
"""

import os
import argparse
import cv2
from pathlib import Path
from datetime import datetime
import time
from tqdm import tqdm

def time_to_seconds(time_str):
    """
    Convert time string in format HH:MM:SS or MM:SS to seconds
    
    Args:
        time_str: Time string in format HH:MM:SS or MM:SS
    
    Returns:
        float: Time in seconds
    """
    # Try to parse as HH:MM:SS
    try:
        t = datetime.strptime(time_str, "%H:%M:%S")
        return t.hour * 3600 + t.minute * 60 + t.second
    except ValueError:
        pass
    
    # Try to parse as MM:SS
    try:
        t = datetime.strptime(time_str, "%M:%S")
        return t.minute * 60 + t.second
    except ValueError:
        pass
    
    # Try to parse as seconds (float)
    try:
        return float(time_str)
    except ValueError:
        raise ValueError(f"Invalid time format: {time_str}. Use HH:MM:SS, MM:SS, or seconds.")

def extract_frames(video_path, output_dir, start_time, end_time):
    """
    Extract frames from a video file based on specified start and end times
    
    Args:
        video_path: Path to the video file
        output_dir: Directory to save extracted frames
        start_time: Start time in seconds
        end_time: End time in seconds
    
    Returns:
        int: Number of extracted frames
    """
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Open the video file
    video = cv2.VideoCapture(video_path)
    if not video.isOpened():
        raise ValueError(f"Could not open video file: {video_path}")
    
    # Get video properties
    fps = video.get(cv2.CAP_PROP_FPS)
    total_frames = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
    duration = total_frames / fps
    
    print(f"Video properties:")
    print(f"  - FPS: {fps}")
    print(f"  - Total frames: {total_frames}")
    print(f"  - Duration: {duration:.2f} seconds")
    
    # Validate time range
    if start_time < 0:
        start_time = 0
        print(f"Start time adjusted to 0 seconds")
    
    if end_time > duration:
        end_time = duration
        print(f"End time adjusted to {duration:.2f} seconds (video duration)")
    
    if start_time >= end_time:
        raise ValueError(f"Start time ({start_time:.2f}s) must be less than end time ({end_time:.2f}s)")
    
    # Calculate frame indices
    start_frame = int(start_time * fps)
    end_frame = int(end_time * fps)
    
    print(f"Extracting frames from {start_time:.2f}s to {end_time:.2f}s")
    print(f"Frame range: {start_frame} to {end_frame}")
    
    # Set video position to start frame
    video.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
    
    # Extract frames
    frame_count = 0
    frame_index = 0
    
    with tqdm(total=end_frame-start_frame+1, desc="Extracting frames") as pbar:
        while True:
            # Read frame
            ret, frame = video.read()
            
            # Break if end of video or reached end frame
            if not ret or frame_index > (end_frame - start_frame):
                break
            
            # Save frame
            output_path = os.path.join(output_dir, f"frame_{frame_count:04d}.png")
            cv2.imwrite(output_path, frame)
            
            frame_count += 1
            frame_index += 1
            pbar.update(1)
    
    # Release video
    video.release()
    
    print(f"Extracted {frame_count} frames to {output_dir}")
    return frame_count

def main():
    parser = argparse.ArgumentParser(description='Extract frames from a video file')
    
    # Get script directory
    script_dir = os.path.dirname(os.path.abspath(__file__))
    
    # Input/output paths
    parser.add_argument('--video', type=str, required=True,
                        help='Path to the video file')
    parser.add_argument('--output', type=str, default=os.path.join(script_dir, 'images'),
                        help='Directory to save extracted frames')
    
    # Time range
    parser.add_argument('--start', type=str, default='0',
                        help='Start time (HH:MM:SS, MM:SS, or seconds)')
    parser.add_argument('--end', type=str, default='-1',
                        help='End time (HH:MM:SS, MM:SS, or seconds). Use -1 for end of video.')
    
    args = parser.parse_args()
    
    # Convert time strings to seconds
    start_time = time_to_seconds(args.start)
    
    if args.end == '-1':
        # Open video to get duration
        video = cv2.VideoCapture(args.video)
        if not video.isOpened():
            raise ValueError(f"Could not open video file: {args.video}")
        
        fps = video.get(cv2.CAP_PROP_FPS)
        total_frames = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
        duration = total_frames / fps
        video.release()
        
        end_time = duration
    else:
        end_time = time_to_seconds(args.end)
    
    # Extract frames
    extract_frames(args.video, args.output, start_time, end_time)

if __name__ == "__main__":
    main() 