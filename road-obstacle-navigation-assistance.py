#!/usr/bin/env python
# coding: utf-8

# In[2]:


import os
import random
from shutil import copy2

# Paths
labels_dir = r"C:\Users\chitr\Downloads\data_object_label_2\training\label_2"
dataset_images_dir = r"C:\Users\chitr\Downloads\image_2"

# Output YOLO dataset paths
yolo_dataset_path = r"C:\Users\chitr\Downloads\YOLOv8_obstacle_dataset"
yolo_labels_dir = os.path.join(yolo_dataset_path, "labels")
yolo_images_dir = os.path.join(yolo_dataset_path, "images")
os.makedirs(yolo_labels_dir, exist_ok=True)
os.makedirs(yolo_images_dir, exist_ok=True)

# Class Mapping for Road Obstacles
class_mapping = {
    "Car": 0, "Pedestrian": 1, "Cyclist": 2, "Person": 3, 
    "Bicycle": 4, "Motorcycle": 5, "Bus": 6, "Truck": 7, 
    "Traffic_cone": 8, "Pothole": 9, "Barrier": 10
}

# Convert Labels to YOLO
def convert_to_yolo(label_path, output_path, img_width=1242, img_height=375):
    with open(label_path, "r") as file:
        lines = file.readlines()

    yolo_lines = []
    for line in lines:
        data = line.split()
        class_name, x1, y1, x2, y2 = data[0], float(data[4]), float(data[5]), float(data[6]), float(data[7])

        if class_name not in class_mapping:
            continue

        # Convert Bounding Box to YOLO format
        class_id = class_mapping[class_name]
        x_center = ((x1 + x2) / 2) / img_width
        y_center = ((y1 + y2) / 2) / img_height
        width = (x2 - x1) / img_width
        height = (y2 - y1) / img_height

        yolo_lines.append(f"{class_id} {x_center} {y_center} {width} {height}\n")

    with open(output_path, "w") as f:
        f.writelines(yolo_lines)

# Process All Label File
for file in os.listdir(labels_dir):
    convert_to_yolo(
        os.path.join(labels_dir, file),
        os.path.join(yolo_labels_dir, file)
    )

print("labels converted to YOLO format!")


# In[3]:


# Define train-test split ratio
train_ratio = 0.8

# Get all images
all_images = [f for f in os.listdir(dataset_images_dir) if f.endswith(('.jpg', '.png'))]
random.shuffle(all_images)

# Split dataset
train_size = int(len(all_images) * train_ratio)
train_images = all_images[:train_size]
test_images = all_images[train_size:]

# Create Train and Test Directories
train_image_dir = os.path.join(yolo_dataset_path, "images/train")
train_label_dir = os.path.join(yolo_dataset_path, "labels/train")
test_image_dir = os.path.join(yolo_dataset_path, "images/test")
test_label_dir = os.path.join(yolo_dataset_path, "labels/test")

os.makedirs(train_image_dir, exist_ok=True)
os.makedirs(train_label_dir, exist_ok=True)
os.makedirs(test_image_dir, exist_ok=True)
os.makedirs(test_label_dir, exist_ok=True)

# Move Images and Labels
for img in train_images:
    copy2(os.path.join(dataset_images_dir, img), os.path.join(train_image_dir, img))
    label_name = img.replace(".jpg", ".txt").replace(".png", ".txt")
    if os.path.exists(os.path.join(yolo_labels_dir, label_name)):
        copy2(os.path.join(yolo_labels_dir, label_name), os.path.join(train_label_dir, label_name))

for img in test_images:
    copy2(os.path.join(dataset_images_dir, img), os.path.join(test_image_dir, img))
    label_name = img.replace(".jpg", ".txt").replace(".png", ".txt")
    if os.path.exists(os.path.join(yolo_labels_dir, label_name)):
        copy2(os.path.join(yolo_labels_dir, label_name), os.path.join(test_label_dir, label_name))

print("Dataset successfully split into train and test sets!")


# In[6]:


from ultralytics import YOLO
import torch

# Check if GPU is available
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")

# Load YOLO model (using the smaller 'nano' version for faster training)
model = YOLO("yolov8n.pt")  # Try "yolov8s.pt" if you need better accuracy

# Train the model with optimized settings
model.train(
    data="C:/Users/chitr/Downloads/YOLOv8_obstacle_dataset/dataset.yaml",  # Path to dataset
    epochs=5,            # Further reduced epochs for faster training
    imgsz=416,           # Lower image size speeds up training
    batch=32,            # Increased batch size for better parallel processing
    device=device,       # Use GPU if available
    amp=True,            # Enable Automatic Mixed Precision for efficiency
    workers=4,           # Number of CPU workers for data loading
    patience=2,          # Stop early if no improvement
    cache=True           # Caches dataset for faster training
)

print("YOLOv8 training complete!")


# In[7]:


import os
import cv2
from ultralytics import YOLO

# Load trained YOLO model
model = YOLO("C:/Users/chitr/runs/detect/train35/weights/last.pt")

# Define test images path
test_image_dir = r"C:\Users\chitr\Downloads\YOLOv8_obstacle_dataset\images\test"
output_dir = r"C:\Users\chitr\Downloads\YOLOv8_obstacle_dataset\detection"
os.makedirs(output_dir, exist_ok=True)

# Detect obstacles in test images
for img_name in os.listdir(test_image_dir):
    img_path = os.path.join(test_image_dir, img_name)
    
    results = model(img_path)  # Run YOLO detection
    
    for i, result in enumerate(results):  # Iterate over result list
        save_path = os.path.join(output_dir, f"detection_{img_name}")  # Save path
        result.save(filename=save_path)  # Save image with detections

print("Detection complete! Results saved in 'detection' folder.")



# In[8]:


import cv2
import numpy as np
import IPython.display as display
from PIL import Image
from ultralytics import YOLO
import pyttsx3

# Initialize Text-to-Speech Engine
engine = pyttsx3.init()
engine.setProperty('rate', 150)

# Load YOLOv8 Model
model = YOLO("yolov8n.pt")

# Define input image path
input_image_path = r"C:\Users\chitr\Downloads\YOLOv8_obstacle_dataset\images\test\001256.png"
# Object Priority Dictionary
object_priority = {
    "Car": 0, "Pedestrian": 1, "Cyclist": 2, "Person": 3, 
    "Bicycle": 4, "Motorcycle": 5, "Bus": 6, "Truck": 7, 
    "Traffic_cone": 8, "Pothole": 9, "Barrier": 10
}

# Function to estimate distance based on bounding box height
def estimate_distance(box_height, known_height=1.5, focal_length=800):
    """Estimates object distance in meters using a simple depth approximation."""
    return round((known_height * focal_length) / box_height, 1)

# Function to detect only the closest obstacle for turning
def detect_turning_obstacle(image_path):
    img = cv2.imread(image_path)
    results = model(img)
    
    width = img.shape[1]
    closest_obstacle = None
    turn_direction = None
    voice_message = ""

    for result in results:
        for box in result.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            cls = int(box.cls[0])
            label = model.names[cls]

            # Estimate distance
            obj_height = y2 - y1
            distance = estimate_distance(obj_height)

            # Determine object priority
            priority = object_priority.get(label, 1)

            # Determine position
            center_x = (x1 + x2) // 2
            position = "left" if center_x < width // 2 else "right"

            # Update closest obstacle for turning decision
            if closest_obstacle is None or distance < closest_obstacle[1]:
                closest_obstacle = (label, distance, position, priority)

    # Decide turning direction and generate voice message
    if closest_obstacle:
        obj, dist, pos, pri = closest_obstacle
        if pos == "left":
            turn_direction = f"CRITICAL: {obj} ({dist}m) on the LEFT. Move RIGHT."
            voice_message = f"Warning! {obj} detected at {dist} meters on the left. Move right immediately."
        elif pos == "right":
            turn_direction = f"CRITICAL: {obj} ({dist}m) on the RIGHT. Move LEFT."
            voice_message = f"Warning! {obj} detected at {dist} meters on the right. Move left immediately."
    else:
        turn_direction = "No obstacles requiring a turn detected."
        voice_message = "No obstacles detected. Continue driving safely."

    return turn_direction, img, voice_message

# Function to play voice alerts
def voice_alert(text):
    engine.say(text)
    engine.runAndWait()

# Run Detection
turning_decision, processed_img, voice_message = detect_turning_obstacle(input_image_path)

# Speak the alert
voice_alert(voice_message)

# Display Results in Jupyter
print(" Road Obstacle Detection Result")
print(turning_decision)

# Show Processed Image in Jupyter Notebook
display.display(Image.fromarray(cv2.cvtColor(processed_img, cv2.COLOR_BGR2RGB)))


# In[12]:


import cv2
import numpy as np
import IPython.display as display
from PIL import Image
from ultralytics import YOLO
import pyttsx3

# Initialize Text-to-Speech Engine
engine = pyttsx3.init()
engine.setProperty('rate', 150)

# Load YOLOv8 Model
model = YOLO("yolov8n.pt")

# Define input image path
input_image_path = r"C:\Users\chitr\Downloads\YOLOv8_obstacle_dataset\images\test\002173.png"

# Object Priority Dictionary
object_priority = {
    "Car": 0, "Pedestrian": 1, "Cyclist": 2, "Person": 3, 
    "Bicycle": 4, "Motorcycle": 5, "Bus": 6, "Truck": 7, 
    "Traffic_cone": 8, "Pothole": 9, "Barrier": 10
}

# Function to estimate distance based on bounding box height
def estimate_distance(box_height, known_height=1.5, focal_length=800):
    """Estimates object distance in meters using a simple depth approximation."""
    return round((known_height * focal_length) / box_height, 1)

# Function to detect only the closest obstacle for turning
def detect_turning_obstacle(image_path):
    img = cv2.imread(image_path)
    results = model(img)
    
    width = img.shape[1]
    closest_obstacle = None
    turn_direction = None
    voice_message = ""

    for result in results:
        for box in result.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            cls = int(box.cls[0])
            label = model.names[cls]

            # Estimate distance
            obj_height = y2 - y1
            distance = estimate_distance(obj_height)

            # Determine object priority
            priority = object_priority.get(label, 1)

            # Determine position
            center_x = (x1 + x2) // 2
            position = "left" if center_x < width // 2 else "right"

            # Update closest obstacle for turning decision
            if closest_obstacle is None or distance < closest_obstacle[1]:
                closest_obstacle = (label, distance, position, priority)

    # Decide turning direction and generate voice message
    if closest_obstacle:
        obj, dist, pos, pri = closest_obstacle
        if pos == "left":
            turn_direction = f" CRITICAL: {obj} ({dist}m) on the LEFT. Move RIGHT."
            voice_message = f"Warning! {obj} detected at {dist} meters on the left. Move right immediately."
        elif pos == "right":
            turn_direction = f" CRITICAL: {obj} ({dist}m) on the RIGHT. Move LEFT."
            voice_message = f"Warning! {obj} detected at {dist} meters on the right. Move left immediately."
    else:
        turn_direction = "No obstacles requiring a turn detected."
        voice_message = "No obstacles detected. Continue driving safely."

    return turn_direction, img, voice_message

# Function to play voice alerts
def voice_alert(text):
    engine.say(text)
    engine.runAndWait()

# Run Detection
turning_decision, processed_img, voice_message = detect_turning_obstacle(input_image_path)

# Speak the alert
voice_alert(voice_message)

# Display Results in Jupyter
print("Road Obstacle Detection Result")
print(turning_decision)

# Show Processed Image in Jupyter Notebook
display.display(Image.fromarray(cv2.cvtColor(processed_img, cv2.COLOR_BGR2RGB)))


# In[1]:


import os
import cv2
import numpy as np
import pyttsx3
from ultralytics import YOLO

# Initialize Text-to-Speech Engine
engine = pyttsx3.init()
engine.setProperty('rate', 150)

# Load YOLOv8 Model
model = YOLO("yolov8n.pt")  # You can change this to yolov8s.pt, yolov8m.pt, etc.

# Define input and output video paths
input_video_path = r"C:\Users\chitr\Downloads\YOLOv8_obstacle_dataset\testing_video.mp4"
output_video_path = r"C:\Users\chitr\Downloads\YOLOv8_obstacle_dataset\output_video.mp4"
# Object Priority Dictionary
object_priority = {
    "Car": 0, "Pedestrian": 1, "Cyclist": 2, "Person": 3, 
    "Bicycle": 4, "Motorcycle": 5, "Bus": 6, "Truck": 7, 
    "Traffic_cone": 8, "Pothole": 9, "Barrier": 10
}

# Distance threshold for voice alerts
alert_distance = 3  # Only alert when objects are closer than this (in meters)

# Function to estimate distance based on bounding box height
def estimate_distance(box_height, known_height=1.5, focal_length=800):
    """Estimates object distance in meters using a simple depth approximation."""
    return round((known_height * focal_length) / box_height, 1)

# Process the video
cap = cv2.VideoCapture(input_video_path)

# Get video properties
frame_width = int(cap.get(3))
frame_height = int(cap.get(4))
fps = int(cap.get(cv2.CAP_PROP_FPS))

# Define Video Writer
fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Codec for .mp4
out = cv2.VideoWriter(output_video_path, fourcc, fps, (frame_width, frame_height))

# Process frames
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    results = model(frame)

    width = frame.shape[1]
    closest_obstacle = None
    voice_message = ""

    for result in results:
        for box in result.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            cls = int(box.cls[0])
            label = model.names[cls]

            # Estimate distance
            obj_height = y2 - y1
            distance = estimate_distance(obj_height)

            # Determine object priority
            priority = object_priority.get(label, 1)

            # Determine position
            center_x = (x1 + x2) // 2
            position = "left" if center_x < width // 2 else "right"

            # Draw bounding box and label
            color = (0, 255, 0) if distance > alert_distance else (0, 0, 255)  # Green for far, Red for close
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 3)
            cv2.putText(frame, f"{label} {distance}m", (x1, y1 - 10), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

            # Update closest obstacle for turning decision
            if closest_obstacle is None or distance < closest_obstacle[1]:
                closest_obstacle = (label, distance, position, priority)

    # Decide turning direction and generate voice message **only if the object is very close**
    if closest_obstacle:
        obj, dist, pos, pri = closest_obstacle
        if dist <= alert_distance:  # **Trigger voice alert only for close obstacles**
            if pos == "left":
                voice_message = f"Warning! {obj} very close at {dist} meters on the left. Move right!"
            elif pos == "right":
                voice_message = f"Warning! {obj} very close at {dist} meters on the right. Move left!"

    # Speak the alert **only if a new message is generated**
    if voice_message:
        engine.say(voice_message)
        engine.runAndWait()

    # Write the frame to the output video
    out.write(frame)

    # Display the frame (Optional)
    cv2.imshow("Obstacle Detection", frame)

    # Press 'q' to exit early
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release resources
cap.release()
out.release()
cv2.destroyAllWindows()

print(f"\n Video Processing Completed! Saved as {output_video_path}")


# In[1]:


# In[1]:


import cv2
import os
import numpy as np
from ultralytics import YOLO

# Load YOLO model
model = YOLO("yolov8n.pt")  # Use 'yolov8s.pt' for better accuracy

# Paths
image_folder = r"C:\Users\chitr\Downloads\YOLOv8_obstacle_dataset\images\test"
pred_folder = r"C:\Users\chitr\Downloads\YOLOv8_obstacle_dataset\images\test_pred"
os.makedirs(pred_folder, exist_ok=True)

# Precision & Recall Estimation Variables
total_detections = 0
high_conf_detections = 0
image_count = 0
iou_threshold = 0.5  # IOU threshold for duplicate filtering

# Function to calculate Intersection over Union (IoU)
def iou(box1, box2):
    x1, y1, x2, y2 = box1
    x1g, y1g, x2g, y2g = box2

    # Compute intersection
    xi1, yi1, xi2, yi2 = max(x1, x1g), max(y1, y1g), min(x2, x2g), min(y2, y2g)
    inter_area = max(0, xi2 - xi1) * max(0, yi2 - y1)

    # Compute union
    box1_area = (x2 - x1) * (y2 - y1)
    box2_area = (x2g - x1g) * (y2g - y1g)
    union_area = box1_area + box2_area - inter_area

    # Compute IoU
    return inter_area / union_area if union_area != 0 else 0

# Run detection on images
for image_name in os.listdir(image_folder):
    image_path = os.path.join(image_folder, image_name)
    image = cv2.imread(image_path)

    if image is None:
        continue

    results = model(image)
    image_count += 1

    detections = []
    for result in results:
        for box in result.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])  # Bounding box
            conf = box.conf[0].item()  # Confidence score
            cls = int(box.cls[0])  # Class ID
            detections.append((x1, y1, x2, y2, conf, cls))

    # Sort detections by confidence (high to low)
    detections = sorted(detections, key=lambda x: x[4], reverse=True)

    filtered_detections = []
    for det in detections:
        if all(iou(det[:4], fd[:4]) < iou_threshold for fd in filtered_detections):
            filtered_detections.append(det)

    # Update metrics
    total_detections += len(detections)
    high_conf_detections += len(filtered_detections)

    # Draw bounding boxes
    for (x1, y1, x2, y2, conf, cls) in filtered_detections:
        label = f"{model.names[cls]}: {conf:.2f}"
        cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(image, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    # Save the processed image as PNG
    output_path = os.path.join(pred_folder, image_name.replace('.jpg', '.png').replace('.jpeg', '.png'))
    cv2.imwrite(output_path, image)

# Compute estimated precision, recall, and accuracy
precision = high_conf_detections / total_detections if total_detections > 0 else 0
recall = high_conf_detections / (high_conf_detections + (total_detections - high_conf_detections)) if total_detections > 0 else 0

# Estimate accuracy (assumes high-confidence detections are true positives)
accuracy = high_conf_detections / total_detections if total_detections > 0 else 0

# Print final evaluation results
print("\nEstimated Metrics Without Ground Truth:")
print(f"Total Images Processed: {image_count}")
print(f"Total Objects Detected: {total_detections}")
print(f"High Confidence Detections (Filtered): {high_conf_detections}")
print(f"Estimated Precision: {precision:.2f}")
print(f"Estimated Recall: {recall:.2f}")
print(f"Estimated Accuracy: {accuracy:.2f}")


# In[ ]:




