# Lunar-crater-detection
Note: The .pt model files are not included in this repository due to size limits. Please download them individually before running the app.
Setup Instructions:
1. Clone the repository.
2. Install the requirements:
   `pip install -r requirements.txt`
3. **Download the YOLO Models manually:**
   Because of GitHub's file size limits, the model weights are not included. 
   Please download the following models and place them in the **root directory** (same folder as `app.py`):
   - [yolov5s.pt](https://github.com/ultralytics/assets/releases/download/v8.2.0/yolov5s.pt)
   - [yolov8s.pt](https://github.com/ultralytics/assets/releases/download/v8.2.0/yolov8s.pt)
   - [yolo11n.pt](https://github.com/ultralytics/assets/releases/download/v8.2.0/yolo11n.pt)
   
   (Note: If your filenames are different, update the "Model Paths" in the app's sidebar to match your file names).
4. Run the app:
   `streamlit run app.py`
