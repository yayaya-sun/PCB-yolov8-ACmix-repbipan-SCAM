# PCB-yolov8-ACmix-repbipan-SCAM（The Visual Computer）
This is the implementation of the paper: "Enhancing Micro-Scale PCB Defect Detection through Hybrid Attention and Multi-Scale Feature Fusion"

## Environment
**The code is tested on:**

 - PyTorch  1.11.0
 - Python  3.8(ubuntu20.04)
 - CUDA  11.3

**Create a virtual environment and activate it.**
```bash
conda create -n name python=3.8
conda activate name
```
**install Dependencies**
```bash
pip install torch==1.13.0+cu116 torchvision==0.14.0+cu116 torchaudio==0.13.0 -c pytorch -c nvidia
pip install -r requirements.txt
```

## Dataset Preparation

The public dataset is available for download at the following link:[PKU-Market-PCB](https://modelscope.cn/datasets/OmniData/PKU-Market-PCB).

(1)Using ```bash ultralytics/DA.py``` for data augmentation on original images and corresponding XML annotation files.

(2)Using ```bash ultralytics/xmlConvertTxt.py``` to convert the augmented XML files into YOLO standard format (TXT files).

(3)Splitting the dataset using cross-validation.

## Train
```bash
python ultralytics/train.py
```
This file needs to be modified:

(1) Line 8: The ```bash YAML``` file of the model to be trained.

(2) Line 9: Load the pre-trained model (downloaded from the official website).

(3) Line 13: The dataset YAML file.

(4) After Line 13: Training epochs, patience, batch size, optimizer selection, learning rate, etc. (Choose flexibly based on your computer's configuration and training requirements).

## Evaluation

```bash
python ultralytics/val.py
```
DOI： 10.5281/zenodo.18058043
