"""
Video windowing script
======================
This script processes videos in the input directory and splits them into windows of a specified size. The windows are saved
as separate video files in the output directory.

Input:
- The input directory should contain video files in MP4 format.

Output:
- The output directory will contain the processed video windows.

Parameters:
- window_size: Size of the window in seconds (default: 20 seconds)
- step: Step size in seconds for the sliding window (default: 5 seconds)

Example usage:
python 4_dataWindowing.py --window_size 20 --step 5
"""

# Libraries -----------------------------------------------------
from moviepy.editor import VideoFileClip
import os
import argparse

# Argument parsing -----------------------------------------------------
# Configurar el analizador de argumentos
parser = argparse.ArgumentParser(description="Process videos into windows.")
parser.add_argument("--window_size", type=int, default=20, help="Size of the window in seconds")
parser.add_argument("--step", type=int, default=5, help="Step size in seconds for the sliding window")

# Leer argumentos de la línea de comando
args = parser.parse_args()

# Paths ---------------------------------------------------------
# Directorios de trabajo
input_directory = '../data/processed'
output_directory = '../data/windows'

# Crear el directorio de salida si no existe
if not os.path.exists(output_directory):
    os.makedirs(output_directory)

# Functions ---------------------------------------------------
# Función para procesar cada video
def process_video(video_path, window_size, step):
    """
    Process a video by splitting it into windows of a specified size and step.

    Args:
        video_path (str): The path to the video file.
        window_size (float): The duration (in seconds) of each window.
        step (float): The duration (in seconds) between the start of each window.

    Returns:
        None
    """
    clip = VideoFileClip(video_path)
    duration = clip.duration
    
    current_start = 0
    window_count = 0

    while current_start + window_size <= duration:
        window_end = current_start + window_size
        output_path = os.path.join(output_directory, f"{os.path.basename(video_path)}_window_{window_count}.mp4")
        subclip = clip.subclip(current_start, window_end)
        subclip.write_videofile(output_path, codec="libx264")
        window_count += 1
        current_start += step

    # Última ventana si no llega a 20 segundos
    if current_start < duration:
        output_path = os.path.join(output_directory, f"{os.path.basename(video_path)}_window_{window_count}.mp4")
        subclip = clip.subclip(current_start, duration)
        subclip.write_videofile(output_path, codec="libx264")

# Application ---------------------------------------------------
# Procesar todos los archivos en el directorio de entrada
for filename in os.listdir(input_directory):
    if filename.endswith(".mp4"):
        process_video(os.path.join(input_directory, filename), args.window_size, args.step)
