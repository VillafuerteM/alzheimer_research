# libraries ----
import os
from moviepy.editor import VideoFileClip

# function to convert from avi to mp4 ----
def convert_avi_to_mp4(input_path, output_path):
    clip = VideoFileClip(input_path)
    clip.write_videofile(output_path, codec='libx264')

# function to convert all avi files in a folder ----
def convert_avi_to_mp4_folder(input_folder, output_folder):
    files = os.listdir(input_folder)
    for file in files:
        if file.endswith('.avi'):
            input_path = input_folder + file
            output_path = output_folder + file.replace('.avi', '.mp4')
            convert_avi_to_mp4(input_path, output_path)

# input and output folders
output_folder = '../data/processed/'
input_folder = '../data/raw/Fase inicial/'
convert_avi_to_mp4_folder(input_folder, output_folder)

# we repeat for all subfolders of raw data
input_folder = '../data/raw/Fase intermedia/'
convert_avi_to_mp4_folder(input_folder, output_folder)

input_folder = '../data/raw/Fase inicial B/'
convert_avi_to_mp4_folder(input_folder, output_folder)

input_folder = '../data/raw/Fase Avanzada a/'
convert_avi_to_mp4_folder(input_folder, output_folder)