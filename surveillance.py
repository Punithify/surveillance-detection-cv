import cv2
import torch
import numpy as np
import os
import argparse
from datetime import datetime

# Initialize YOLOv5 model
model = torch.hub.load('ultralytics/yolov5', 'yolov5s')  # Load YOLOv5 small model

# Define a dictionary for mapping object names to COCO class indices
OBJECT_CLASSES = {
    "person": 0,
    "cat": 16,
    "dog": 17
}

def process_video(input_video_path, output_folder, interested_classes):
    # Open video stream or file
    cap = cv2.VideoCapture(input_video_path)
    
    # Check if video was opened correctly
    if not cap.isOpened():
        print("Error: Could not open video stream or file.")
        return

    # Get the video frame width and height
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    # Define codec and create VideoWriter object for recording (this will record the whole video)
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    
    # Create a placeholder for the output path, will be generated only if object detected
    output_path = None
    out = None

    # Flag to track if the object was detected
    object_detected = False
    is_recording = False
    frames_since_last_detection = 0
    max_no_detection_frames = 30  # Stop recording after 30 frames without detection

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Perform object detection
        results = model(frame)
        
        # Parse detected objects
        detections = results.pandas().xywh[0]  # Get pandas dataframe of detected objects

        # Debugging: Print columns to understand the structure
        print("Detection columns:", detections.columns)

        # Handle bounding box extraction with correct column names
        detected_boxes = detections[['xcenter', 'ycenter', 'width', 'height']].values  # Corrected column names
        detected_classes = detections['class'].values
        detected_labels = detections['name'].values

        # Check if any of the detected objects are in our interested classes
        record_this_frame = False
        for i, detected_class in enumerate(detected_classes):
            if detected_class in interested_classes:
                record_this_frame = True
                label = detected_labels[i]
                box = detected_boxes[i]

                # Convert center, width, height to xmin, ymin, xmax, ymax
                xcenter, ycenter, width, height = map(int, box)
                xmin = int(xcenter - width / 2)
                ymin = int(ycenter - height / 2)
                xmax = int(xcenter + width / 2)
                ymax = int(ycenter + height / 2)

                # Draw bounding box and label
                cv2.rectangle(frame, (xmin, ymin), (xmax, ymax), (0, 255, 0), 2)  # Green box
                cv2.putText(frame, label, (xmin, ymin - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)  # Green label

                print(f"Detected {label} at frame {cap.get(cv2.CAP_PROP_POS_FRAMES)}")
                object_detected = True

        # If an interesting object is detected, start recording
        if record_this_frame:
            if not is_recording:
                # Start recording when the first object is detected
                is_recording = True
                # Set the output path and create the VideoWriter object
                timestamp = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
                output_path = os.path.join(output_folder, f'recorded_{timestamp}.avi')
                out = cv2.VideoWriter(output_path, fourcc, 20.0, (frame_width, frame_height))
                print(f"Started recording to {output_path}")
            
            # Write the current frame to the output file
            out.write(frame)
            frames_since_last_detection = 0  # Reset the frame counter after recording a detected frame
        else:
            frames_since_last_detection += 1
        
        # If recording and no object is detected for a few frames, stop recording
        if is_recording and frames_since_last_detection > max_no_detection_frames:
            is_recording = False
            print("Stopped recording due to no object detection.")

        # Display the frame
        cv2.imshow('Frame', frame)
        
        # Break the loop if the user presses 'q'
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()

    # If recording was started, finalize the video
    if out is not None:
        out.release()
    
    cv2.destroyAllWindows()

    # Check if no objects were detected and log it
    if not object_detected:
        print(f"Log: The specified object ({list(OBJECT_CLASSES.keys())[interested_classes[0]]}) was not detected in the video.")
    else:
        print(f"Recording completed and saved to {output_path}.")

def parse_args():
    # Command line argument parsing
    parser = argparse.ArgumentParser(description="Object detection video processing")
    
    # Add arguments for the input video, output folder, and object to search
    parser.add_argument("input_video", type=str, help="Path to the input video file or stream")
    parser.add_argument("output_folder", type=str, help="Folder to save the recorded video")
    parser.add_argument("object", type=str, choices=OBJECT_CLASSES.keys(), help="Object to detect (person, cat, dog, etc.)")
    
    return parser.parse_args()

if __name__ == "__main__":
    # Parse command line arguments
    args = parse_args()
    
    # Get the class index based on the object name passed from the command line
    interested_classes = [OBJECT_CLASSES[args.object]]
    
    # Create output directory if it doesn't exist
    if not os.path.exists(args.output_folder):
        os.makedirs(args.output_folder)

    # Process the video and record when needed
    process_video(args.input_video, args.output_folder, interested_classes)
