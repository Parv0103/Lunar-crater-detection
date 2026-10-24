# Lunar-crater-detection

A Streamlit app that runs pretrained YOLO models on an uploaded lunar surface image and presents the detections with several comparison views.

Important: the app uses generic pretrained YOLO weights (YOLOv5s, YOLOv8s, YOLO11n). They were not trained on a lunar crater dataset. The comparison and performance numbers in the "Model Comparison" and "Performance Metrics" views are hardcoded placeholder values, not measured results.

What is real and what is a placeholder
Part	Status
Running a YOLO model on an uploaded image (confidence, IoU, image size, tile size and augmentation settings)	Real: uses the Ultralytics library with the weights you provide
Detections drawn on your image	Real output of the pretrained model. Because the weights are not crater-specific, treat the boxes as a demonstration of the pipeline, not as validated crater detection
Model comparison table and charts (detections, inference time, model size, accuracy, precision, recall, radar and bar charts)	Hardcoded. Defined in the model_results dictionary in app.py. The numbers were typed in; they were not produced by evaluating the models on a labelled crater dataset
"Custom Trained" row (a fine-tuned YOLOv8)	Placeholder. No custom-trained weights are included in this repository, and the figures shown for it are not measured
Text such as "Best Efficiency" and "Domain-Specific"	Static descriptive text based on those hardcoded numbers

Do not cite any accuracy, precision or recall figure from this app as a measured result.

Setup

The model weights are not stored in this repository (file size). Download them and place them in the same folder as app.py:

yolov5s.pt
yolov8s.pt
yolo11n.pt

If your filenames differ, change the "Model Paths" in the sidebar.

bash
pip install streamlit ultralytics torch opencv-python numpy pandas matplotlib seaborn plotly pillow
streamlit run app.py

The requirements.txt in this repository does not list Streamlit, Ultralytics, PyTorch or Plotly, which the app imports, and it pins older TensorFlow packages the app does not use. Use the install line above until the file is cleaned up.

Views
1.Detection Results
2.Model Comparison (hardcoded numbers)
3.Performance Metrics (hardcoded numbers)
4.Heatmap Visualization
5.Results Summary (export)

Roadmap to a real crater detector (not implemented)
1.Collect or download a labelled lunar crater dataset with bounding boxes.
2.Fine-tune a YOLO model on it with a train/validation/test split.
3.Compute precision, recall and mAP on the held-out test set.
4.Replace the hardcoded model_results with the measured numbers.

Tech stack
Python, Streamlit, Ultralytics YOLO, PyTorch, OpenCV, Plotly.

