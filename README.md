# Image Captioning with CNN + Transformer 

A deep learning model that generates human-like captions for input images using **visual feature extraction (CNN)** and **sequence modeling (Transformer/LSTM)**.

## Problem Statement
Build an end-to-end model that:
1. Extracts visual features from images using a **pretrained CNN** (ResNet50/InceptionV3).
2. Generates contextual captions using a **Transformer Decoder** (or LSTM as baseline).

##  Features
- Supports **Flickr8k/Flickr30k/MS COCO** datasets
- Two model architectures:
  - **CNN + LSTM** (Baseline)
  - **CNN + Transformer** (Advanced)
- Customizable training (transfer learning, attention mechanisms)
- Streamlit web app for demo

##Installation
```bash
git clone https://github.com/Prarabdha14/Caption_app.git
pip install -r requirements.txt
