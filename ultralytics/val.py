import sys
sys.path.append("/root/yolov8")
from ultralytics import YOLO

if __name__ == '__main__':
    # Load a model
    #model = YOLO(r'/mnt/sda/sda1/xwn/runs_yolov8/train/exp_DualAttention/weights/best.pt')   # 使用自己训练好的模型进行评价 # build a new model from YAML
    model = YOLO(r'/root/autodl-tmp/project/runs/train/open-enhance/exp_yolovn8/weights/best.pt')   # 使用自己训练好的模型进行评价 # build a new model from YAML

    #model = YOLO(r'D:\pycharm\ultralytics-main-source\datasets\best.pt');
    # Validate the model
    model.val(
        val=True,  # (bool) validate/test during training
        data=r'/root/autodl-tmp/project/ultralytics/cfg/datasets/PCB.yaml',
        split='test',  # (str) dataset split to use for validation, i.e. 'val', 'test' or 'train' #看测试集的结果
        batch=1,  # (int) number of images per batch (-1 for AutoBatch) #测试的时候设为1是严谨的
        imgsz=640,  # (int) size of input images as integer or w,h
        device='0' ,# (int str  list, optional) device to run on, i.e. cuda device=g or device=0,1,2,3 or device=cpu
        workers=8,  # (int) number of worker threads for data loading (per RANK if DDP)
        save_json = False,  # (bool) save results to JSON file
        save_hybrid=False,  # (bool) save hybrid version of labels (labels + additional predictions)
        conf=0.001,# (float, optional) object confidence threshold for detection (default 0.25 predict, 0.001 val)
        iou=0.5,  # (float) intersection over union (Iou) threshold for NMS
        project='/root/autodl-tmp/project/ultralytics/runs/open-enhance',  # (str, optional) project name
        name='exp_yolov8n-pcb-ACmix',  # (str, optional) experiment name, results saved to project/name' directory
        max_det = 300 ,  # (int) maximum number of detections per image
        half=False,  # (bool) use half precision (FP16)
        dnn=False,  # (bool) use OpenCV DNN for ONNX inference
        plots=True,  # (bool) save plots during train/val
    )

