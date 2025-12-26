# PCB-yolov8-ACmix-repbipan-SCAM
## Dataset Preparation

The public dataset is available for download at the following link:[PKU-Market-PCB](https://modelscope.cn/datasets/OmniData/PKU-Market-PCB).

(1)Using ```bashultralytics/DA.py``` for data augmentation on original images and corresponding XML annotation files.

(2)Using ultralytics/xmlConvertTxt.py to convert the augmented XML files into YOLO standard format (TXT files).

(3)Splitting the dataset using cross-validation.

## Train
```bash
ultralytics/train.py

This file needs to be modified:

(1) Line 8: The YAML file of the model to be trained.

(2) Line 9: Load the pre-trained model (downloaded from the official website).

(3) Line 13: The dataset YAML file.

(4) After Line 13: Training epochs, patience, batch size, optimizer selection, learning rate, etc. (Choose flexibly based on your computer's configuration and training requirements).

## Evaluation

```bash
ultralytics/val.py
```
