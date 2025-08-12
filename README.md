# Road Obstacle Detection and Navigation Assistance using Deep Learning

This project implements a real-time **road obstacle detection and navigation assistance system** using the **YOLOv8 object detection model**. It includes automated voice alerts suggesting left/right navigation based on obstacle type, position, and proximity. The system supports preprocessing, training, evaluation, and full video/image testing.

## 🚦 Project Highlights

- Dataset conversion from KITTI to YOLO format
- Real-time object detection with voice alerts
- Obstacle distance estimation for navigation decisions
- Evaluation using precision, recall, and accuracy 
- Full video and image processing using OpenCV

## ⚙️ Technologies Used

- Python
- OpenCV
- YOLOv8 (Ultralytics)
- pyttsx3 (Text-to-Speech)
- NumPy, Pandas, Matplotlib

## 📦 Requirements

See `requirements.txt`:
```
ultralytics
opencv-python
pyttsx3
numpy
pandas
matplotlib
```

## 🚀 How to Run

1. Clone the repository:
```bash
git clone https://github.com/chitra-cp/road-obstacle-navigation-assistance.git
cd road-obstacle-navigation-assistance
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Run the main script:
```bash
python road-obstacle-navigation-assistance.py
```
- For image or video testing
- For model training and evaluation

## 📊 Output

- Annotated output on images and videos
- Voice alerts like:  
  _“Warning! Car detected on left. Move right!”_
- Summary of:
  - Total detections
  - Estimated precision, recall, accuracy

## 👩‍💻 Author

**Chitra S**  
_M.Sc. Data Analytics Graduate_
