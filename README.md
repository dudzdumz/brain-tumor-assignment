# Brain Tumor MRI Classification using Deep Learning

This project applies deep learning to a real-world binary image classification problem: detecting whether a brain MRI image shows a tumor or no tumor.

The system compares two models:

1. A custom Convolutional Neural Network (CNN) built from scratch.
2. A MobileNetV2 transfer learning model pre-trained on ImageNet.

The project was developed for the Artificial Intelligence and Deep Learning (COMP 20037) individual assignment.

## Dataset

The dataset used is the Brain MRI Images for Brain Tumor Detection dataset from Kaggle.

The dataset contains two classes:

- `yes` = MRI images with tumor
- `no` = MRI images without tumor

Dataset size used:

- No Tumor images: 98
- Tumor images: 155
- Total images: 253

The images were resized to 128x128 pixels and normalized by scaling pixel values to the range 0 to 1.

## Project Structure

```text
brain_tumor_assignment/
│
├── dataset/
│   ├── no/
│   └── yes/
│
├── outputs/
│   ├── custom_cnn_training_graph.png
│   ├── mobilenet_training_graph.png
│   └── fullterminaloutput.txt
│
├── train_brain_tumor.py
├── requirements.txt
└── README.md