"""
Face Detection and Blurring
===========================
This script gets all the videos previously converted to mp4
and applies face detection and blurring to them.
A DNN model is used to detect faces in the video frames.
The detected faces are then blurred using Gaussian blur.
"""
# Librerias -------------------------------------------------------------------
import os
import cv2
import face_recognition
import numpy as np

# Model -----------------------------------------------------------------------
modelFile = "./models/res10_300x300_ssd_iter_140000_fp16.caffemodel"
configFile = "./models/deploy.prototxt"
net = cv2.dnn.readNetFromCaffe(configFile, modelFile)

# Funciones -------------------------------------------------------------------
def detect_faces_dnn(frame, conf_threshold=0.3):
    """
    Detect faces in an image using OpenCV's DNN module with a specified confidence threshold.

    Args:
        frame (np.ndarray): The input image.
        conf_threshold (float): The confidence threshold for detecting faces.

    Returns:
        List[Tuple[int, int, int, int]]: List of bounding box coordinates for detected faces.
    """
    # Convert frame to blob
    blob = cv2.dnn.blobFromImage(frame, 1.0, (300, 300), [104, 117, 123], False, False)
    
    # Set the input blob for the neural network
    net.setInput(blob)
    
    # Get the detections from the network
    detections = net.forward()
    
    face_locations = []
    (h, w) = frame.shape[:2]
    
    # Loop through the detections
    for i in range(detections.shape[2]):
        confidence = detections[0, 0, i, 2]
        
        # Filter out detections with confidence below the threshold
        if confidence > conf_threshold:
            box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
            (startX, startY, endX, endY) = box.astype("int")
            
            # Ensure the bounding box coordinates are within the dimensions of the frame
            startX = max(0, startX)
            startY = max(0, startY)
            endX = min(w, endX)
            endY = min(h, endY)
            
            face_locations.append((startY, endX, endY, startX))
    
    return face_locations

# Paths -----------------------------------------------------------------------
# Videos to blur
video_path = '../data/processed/P2_LONGPRESS.mp4'
# Destination path
output_path = '../data/final/P2_LONGPRESS_TH_04.mp4'

# Main ------------------------------------------------------------------------
video_capture = cv2.VideoCapture(video_path)
fourcc = cv2.VideoWriter_fourcc(*'XVID')
fps = int(video_capture.get(cv2.CAP_PROP_FPS))
frame_width = int(video_capture.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(video_capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
out = cv2.VideoWriter(output_path, fourcc, fps, (frame_width, frame_height))

while video_capture.isOpened():
    ret, frame = video_capture.read()
    if not ret:
        break
    
    # Use a lower threshold for face detection
    face_locations = detect_faces_dnn(frame, conf_threshold=0.45)
    
    for (top, right, bottom, left) in face_locations:
        face_image = frame[top:bottom, left:right]
        blurred_face = cv2.GaussianBlur(face_image, (99, 99), 30)
        frame[top:bottom, left:right] = blurred_face
    
    out.write(frame)

video_capture.release()
out.release()
cv2.destroyAllWindows()


# -----------------------------------------------------------------------------
# Now, let's convert the script into a function and apply it to all the videos in the folder.
# -----------------------------------------------------------------------------
"""
Face Detection and Blurring
===========================
This script gets all the videos previously converted to mp4
and applies face detection and blurring to them.
A DNN model is used to detect faces in the video frames.
The detected faces are then blurred using Gaussian blur.
"""

# Libraries -------------------------------------------------------------------
import os
import cv2
import numpy as np

# Model -----------------------------------------------------------------------
modelFile = "./models/res10_300x300_ssd_iter_140000_fp16.caffemodel"
configFile = "./models/deploy.prototxt"
net = cv2.dnn.readNetFromCaffe(configFile, modelFile)

# Functions -------------------------------------------------------------------
def detect_faces_dnn(frame, conf_threshold=0.3):
    """
    Detect faces in an image using OpenCV's DNN module with a specified confidence threshold.

    Args:
        frame (np.ndarray): The input image.
        conf_threshold (float): The confidence threshold for detecting faces.

    Returns:
        List[Tuple[int, int, int, int]]: List of bounding box coordinates for detected faces.
    """
    # Convert frame to blob
    blob = cv2.dnn.blobFromImage(frame, 1.0, (300, 300), [104, 117, 123], False, False)
    
    # Set the input blob for the neural network
    net.setInput(blob)
    
    # Get the detections from the network
    detections = net.forward()
    
    face_locations = []
    (h, w) = frame.shape[:2]
    
    # Loop through the detections
    for i in range(detections.shape[2]):
        confidence = detections[0, 0, i, 2]
        
        # Filter out detections with confidence below the threshold
        if confidence > conf_threshold:
            box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
            (startX, startY, endX, endY) = box.astype("int")
            
            # Ensure the bounding box coordinates are within the dimensions of the frame
            startX = max(0, startX)
            startY = max(0, startY)
            endX = min(w, endX)
            endY = min(h, endY)
            
            face_locations.append((startY, endX, endY, startX))
    
    return face_locations

def process_video(input_path, output_path, conf_threshold=0.3):
    """
    Apply face detection and blurring to a video.

    Args:
        input_path (str): Path to the input video file.
        output_path (str): Path to save the processed video.
        conf_threshold (float): Confidence threshold for face detection.

    Returns:
        None
    """
    video_capture = cv2.VideoCapture(input_path)
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    fps = int(video_capture.get(cv2.CAP_PROP_FPS))
    frame_width = int(video_capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(video_capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    # Create the output directory if it does not exist
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    out = cv2.VideoWriter(output_path, fourcc, fps, (frame_width, frame_height))

    while video_capture.isOpened():
        ret, frame = video_capture.read()
        if not ret:
            break

        # Use the face detection function
        face_locations = detect_faces_dnn(frame, conf_threshold=conf_threshold)

        for (top, right, bottom, left) in face_locations:
            face_image = frame[top:bottom, left:right]
            blurred_face = cv2.GaussianBlur(face_image, (99, 99), 30)
            frame[top:bottom, left:right] = blurred_face

        out.write(frame)

    video_capture.release()
    out.release()
    print(f"Processed {input_path} and saved to {output_path}")
    cv2.destroyAllWindows()

# Main ------------------------------------------------------------------------
input_folder = '../data/processed/'
output_folder = '../data/final/'

# Ensure the output folder exists
os.makedirs(output_folder, exist_ok=True)

# Iterate over all .mp4 files in the input folder
for root, dirs, files in os.walk(input_folder):
    for file in files:
        if file.endswith('.mp4'):
            input_path = os.path.join(root, file)
            relative_path = os.path.relpath(input_path, input_folder)
            output_path = os.path.join(output_folder, relative_path).replace('.mp4', '_blurred.mp4')
            process_video(input_path, output_path, conf_threshold=0.45)
