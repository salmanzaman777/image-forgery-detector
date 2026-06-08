---
title: Image Forgery Detector
emoji: 🛡️
colorFrom: blue
colorTo: red
sdk: streamlit
sdk_version: 1.35.0
python_version: 3.11
app_file: app.py
pinned: false
---

# Image Forgery Detector

This application detects tampering in images using a Dual-Branch CNN architecture.

## How it works:
1. **RGB Branch:** Uses a pretrained ResNet50 to extract semantic features from the original image.
2. **ELA Branch:** Computes Error Level Analysis (ELA) to detect JPEG compression inconsistencies.
3. **Fused Model:** Combines features from both branches to make a final prediction.

## Explainability:
The app uses **Grad-CAM** to visualize which parts of the image the model focused on when making its decision.

## Deployment:
This app is designed to be deployed on Hugging Face Spaces using Streamlit.

## Repository:
[https://github.com/salmanzaman777/image-forgery-detector](https://github.com/salmanzaman777/image-forgery-detector)

## Documents:
- [Project Report](documents/Project_Report_Digital_Image_Forgery_Detector.docx)
