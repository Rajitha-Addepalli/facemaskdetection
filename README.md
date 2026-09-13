# 😷 Real-Time Face Mask Detection & Compliance Monitoring

A Deep Learning and Computer Vision application that detects whether people are wearing face masks in real-time using a webcam or video file.

The system is designed for **multi-person environments**, providing person tracking, mask/no-mask classification, confidence scores, compliance statistics, violation alerts, logging, and a monitoring dashboard.

## 🚀 Features

- 🧠 **MobileNetV2 Transfer Learning** for mask classification
- 👥 **Multi-person face detection**
- 🎯 Real-time **Mask / No Mask / Uncertain / Occluded** classification
- 📊 Prediction confidence scores
- 🆔 Temporary person tracking with unique IDs
- 📈 Real-time mask compliance percentage
- 🚨 Violation detection and alerts
- 📝 Detection and violation logging
- 📹 Webcam and video-file support
- 📊 Streamlit monitoring dashboard
- ⚡ Batch prediction for improved inference efficiency
- 📋 Model evaluation using:
  - Accuracy
  - Precision
  - Recall
  - F1-score
  - Confusion Matrix

## 🛠 Tech Stack

- Python
- TensorFlow / Keras
- MobileNetV2
- OpenCV
- NumPy
- Scikit-learn
- Streamlit
- Matplotlib

## 🧠 Model Performance

The trained model was evaluated on **1,510 validation images**.

| Metric | Score |
|---|---:|
| Accuracy | 98.87% |
| Precision | 98.87% |
| Recall | 98.87% |
| F1-Score | 98.87% |

### Confusion Matrix

```text
                 Predicted
               With Mask  No Mask
With Mask         737        8
No Mask             9      756