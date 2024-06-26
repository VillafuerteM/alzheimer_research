import cv2
from deepface import DeepFace
import os

# Define the input and output directories
input_dir = '../data/processed2/'
output_dir = '../data/emotion_detection/'

# Create the output directory if it doesn't exist
os.makedirs(output_dir, exist_ok=True)

# List all mp4 files in the input directory
video_files = [f for f in os.listdir(input_dir) if f.endswith('.mp4')]

# Process each video file
for video_file in video_files:
    video_path = os.path.join(input_dir, video_file)
    cap = cv2.VideoCapture(video_path)

    # Determine output path and prepare video writer
    output_path = os.path.join(output_dir, video_file)
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (frame_width, frame_height))

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        try:
            # Detect emotions on the frame
            results = DeepFace.analyze(frame, actions=['emotion'], enforce_detection=False)

            if isinstance(results, list):
                # Multiple faces detected
                for person in results:
                    bounding_box = person["region"]
                    emotions = person["emotion"]

                    # Draw bounding box and annotate emotions
                    x, y, w, h = bounding_box['x'], bounding_box['y'], bounding_box['w'], bounding_box['h']
                    cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 155, 255), 2)
                    y_offset = y + h + 20  # Adjust based on your preference
                    for emotion, score in emotions.items():
                        text = f"{emotion}: {score:.2f}"
                        cv2.putText(frame, text, (x, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 155, 255), 1, cv2.LINE_AA)
                        y_offset += 15
            else:
                # Single face detected
                bounding_box = results["region"]
                emotions = results["emotion"]

                # Draw bounding box and annotate emotions
                x, y, w, h = bounding_box['x'], bounding_box['y'], bounding_box['w'], bounding_box['h']
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 155, 255), 2)
                y_offset = y + h + 105
                for emotion, score in emotions.items():
                    text = f"{emotion}: {score:.2f}"
                    cv2.putText(frame, text, (x, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 155, 255), 1, cv2.LINE_AA)
                    y_offset += 15

        except Exception as e:
            print(f"Error processing frame: {e}")

        # Write the frame to the output video
        out.write(frame)

    # Release resources for the current video
    cap.release()
    out.release()

# Ensure all windows are closed
cv2.destroyAllWindows()
