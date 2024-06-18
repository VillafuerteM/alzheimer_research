"""
Video Converter
================
This script converts all AVI files in a folder to MP4 format.
The input and output folders are defined in the script.

Input:
- AVI files in the input folders.

Output:
- MP4 files in the output folder.
"""

# Libraries -----------------------------------------------------
import os
from moviepy.editor import VideoFileClip

# Paths ---------------------------------------------------------
# output folder - remains the same throughout the script
output_folder = '../data/processed2/'

# input folders - list of folders to convert
input_folders = [
    '../data/raw/Fase inicial/',
    '../data/raw/Fase intermedia/',
    '../data/raw/Fase inicial B/',
    '../data/raw/Fase Avanzada a/'
]

# Functions -----------------------------------------------------
def convert_avi_to_mp4(input_path, output_path):
    """
    Converts an AVI video file to MP4 format.

    Args:
        input_path (str): The path to the input AVI file.
        output_path (str): The path to save the converted MP4 file.

    Raises:
        Exception: If there is an error during the conversion process.

    Returns:
        None
    """
    try:
        clip = VideoFileClip(input_path)
        clip.write_videofile(output_path, codec='libx264')
        print(f"Converted {input_path} to {output_path}")
    except Exception as e:
        print(f"Failed to convert {input_path}: {e}")

def convert_avi_to_mp4_folder(input_folder, output_folder):
    """
    Converts all AVI files in the input folder to MP4 format and saves them in the output folder.

    Args:
        input_folder (str): The path to the folder containing the AVI files.
        output_folder (str): The path to the folder where the converted MP4 files will be saved.

    Returns:
        None
    """
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    for root, _, files in os.walk(input_folder):
        for file in files:
            if file.endswith('.avi'):
                input_path = os.path.join(root, file)
                relative_path = os.path.relpath(input_path, input_folder)
                output_path = os.path.join(output_folder, relative_path).replace('.avi', '.mp4')
                output_dir = os.path.dirname(output_path)
                if not os.path.exists(output_dir):
                    os.makedirs(output_dir)
                convert_avi_to_mp4(input_path, output_path)

# Main Code -----------------------------------------------------
# Convert all AVI files in each input folder
for input_folder in input_folders:
    convert_avi_to_mp4_folder(input_folder, output_folder)
