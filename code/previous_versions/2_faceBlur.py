import os
import cv2
import face_recognition

video_path = '../data/processed/P2_LONGPRESS.mp4'
output_path = '../data/final/P2_LONGPRESS.mp4'

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
    
    rgb_frame = frame[:, :, ::-1]
    
    face_locations = face_recognition.face_locations(rgb_frame)
    
    for (top, right, bottom, left) in face_locations:
        face_image = frame[top:bottom, left:right]
        blurred_face = cv2.GaussianBlur(face_image, (99, 99), 30)
        frame[top:bottom, left:right] = blurred_face
    
    out.write(frame)

video_capture.release()
out.release()
cv2.destroyAllWindows()


# # Main ---------------------------------------------------------
import cv2
import numpy as np

# Load the pre-trained model from OpenCV's DNN module
modelFile = "../models/res10_300x300_ssd_iter_140000_fp16.caffemodel"
configFile = "../models/deploy.prototxt"
net = cv2.dnn.readNetFromCaffe(configFile, modelFile)

def detect_faces_dnn(frame):
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
        
        # Filter out weak detections
        if confidence > 0.5:
            box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
            (startX, startY, endX, endY) = box.astype("int")
            
            # Ensure the bounding box coordinates are within the dimensions of the frame
            startX = max(0, startX)
            startY = max(0, startY)
            endX = min(w, endX)
            endY = min(h, endY)
            
            face_locations.append((startY, endX, endY, startX))
    
    return face_locations

# Video processing with improved face detection
video_path = '../data/processed/P2_LONGPRESS.mp4'
output_path = '../data/final/P2_LONGPRESS.mp4'

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
    
    face_locations = detect_faces_dnn(frame)
    
    for (top, right, bottom, left) in face_locations:
        face_image = frame[top:bottom, left:right]
        blurred_face = cv2.GaussianBlur(face_image, (99, 99), 30)
        frame[top:bottom, left:right] = blurred_face
    
    out.write(frame)

video_capture.release()
out.release()
cv2.destroyAllWindows()


# # Main ---------------------------------------------------------
import cv2
import numpy as np

# Paths to the model files
modelFile = "./models/res10_300x300_ssd_iter_140000_fp16.caffemodel"
configFile = "./models/deploy.prototxt"
net = cv2.dnn.readNetFromCaffe(configFile, modelFile)

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

# Video processing with improved face detection
video_path = '../data/processed/P2_LONGPRESS.mp4'
output_path = '../data/final/P2_LONGPRESS_TH_04.mp4'

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
