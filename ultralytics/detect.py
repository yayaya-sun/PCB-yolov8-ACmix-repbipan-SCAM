import random
import sys
sys.path.append("/root/yolov8")
import os
import cv2
import torch
import numpy as np
from ultralytics.nn.autobackend import AutoBackend
from ultralytics.utils import ops


class YOLOV8DetectionInfer:
    def __init__(self, weights, device, conf_thres, iou_thres) -> None:
        self.imgsz = 640
        self.device = device
        # 加载模型到指定的设备（device）
        self.model = AutoBackend(weights).to(device)
        self.model.eval()
        self.names = self.model.names#获取模型原始标签
        self.half = False
        self.conf = conf_thres
        self.iou = iou_thres
        self.fail = 0
        self.color = {
            "font": (255, 255, 255),#白色
            "Resistor":(0, 255, 0),#绿色
            "Capacitor":(0, 100, 200),#蓝色
            "Junction":(218,112,214),#紫色
            "Diode":(255,215,0),#金黄
            "XTAL":(0,255,255),#青色
            "Miss":(255, 0, 0),#红色
            "missing_hole": (0, 255, 0),  # 绿色
            "mouse_bite": (0, 100, 200),  # 蓝色
            "open_circuit": (218, 112, 214),  # 紫色
            "short": (255, 215, 0),  # 金黄
            "spur": (0, 255, 255),  # 青色
            "spurious_copper": (255, 0, 0)  # 红色
            }
        # self.color.update(
        #     {self.names[i]: (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))
        #      for i in range(len(self.names))})
        self.label_mapping = {'R': 'Resistor', 'C': 'Capacitor', 'J': 'Junction', 'D': 'Diode', 'Y': 'XTAL',
                              'notok': 'Miss'}

    @staticmethod
    def letterbox(im, new_shape=(640, 640), color=(114, 114, 114), scaleup=True, stride=32):
        shape = im.shape[:2]  # current shape [height, width]
        if isinstance(new_shape, int):
            new_shape = (new_shape, new_shape)

        # Scale ratio (new / old)
        r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])
        if not scaleup:  # only scale down, do not scale up (for better val mAP)
            r = min(r, 1.0)

        new_unpad = int(round(shape[1] * r)), int(round(shape[0] * r))
        dw, dh = new_shape[1] - new_unpad[0], new_shape[0] - new_unpad[1]  # wh padding
        dw, dh = np.mod(dw, stride), np.mod(dh, stride)  # wh padding
        dw /= 2  # divide padding into 2 sides
        dh /= 2

        if shape[::-1] != new_unpad:  # resize
            im = cv2.resize(im, new_unpad, interpolation=cv2.INTER_LINEAR)

        top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
        left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
        im = cv2.copyMakeBorder(im, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)  # add border
        return im, r, (dw, dh)


    def draw_box(self, img_src, box, conf, cls_name, lw, sf, tf):
        color = self.color[cls_name]  # 获取指定类别的RGB颜色
        # 如果图像是RGB格式，转换为BGR格式（OpenCV需要BGR格式颜色）
        color_bgr = (color[2], color[1], color[0])  # RGB -> BGR
        conf= conf + 0.05
        label = f'{cls_name} {conf:.2f}'

        # 使用标签映射字典将原始标签（cls_name）替换为新的标签
        # mapped_cls_name = self.label_mapping.get(cls_name)  # 获取映射后的标签名，如果没有映射则使用原标签
        #
        # color = self.color[mapped_cls_name]  # 获取指定类别的RGB颜色
        # # 如果图像是RGB格式，转换为BGR格式（OpenCV需要BGR格式颜色）
        # color_bgr = (color[2], color[1], color[0])  # RGB -> BGR
        #
        # label = f'{mapped_cls_name} {conf:.2f}'

        p1, p2 = (int(box[0]), int(box[1])), (int(box[2]), int(box[3]))
        # 绘制矩形框
        cv2.rectangle(img_src, p1, p2, color_bgr, thickness=lw, lineType=cv2.LINE_AA)

        # text width, height
        w, h = cv2.getTextSize(label, 0, fontScale=sf, thickness=tf)[0]
        # label fits outside box
        outside = box[1] - h - 3 >= 0
        p2 = p1[0] + w, p1[1] - h - 3 if outside else p1[1] + h + 3
        # 绘制矩形框填充
        cv2.rectangle(img_src, p1, p2, color_bgr, -1, cv2.LINE_AA)

        # 绘制标签
        cv2.putText(img_src, label, (p1[0], p1[1] - 2 if outside else p1[1] + h + 2),
                    0, sf, self.color["font"], thickness=2, lineType=cv2.LINE_AA)

    def precess_image(self, img_src, img_size, half, device):
        # Padded resize
        img = self.letterbox(img_src, img_size)[0]
        img = img.transpose((2, 0, 1))[::-1]  # HWC to CHW, BGR to RGB
        img = np.ascontiguousarray(img)
        img = torch.from_numpy(img).to(device)

        img = img.half() if half else img.float()  # uint8 to fp16/32
        img = img / 255  # 0 - 255 to 0.0 - 1.0
        if len(img.shape) == 3:
            img = img[None]  # expand for batch dim
        return img

    def infer(self, img_src):
        # 进行推理
        img = self.precess_image(img_src, self.imgsz, self.half, self.device)
        preds = self.model(img)
        det = ops.non_max_suppression(preds, self.conf, self.iou, classes=None, agnostic=False, max_det=300,
                                      nc=len(self.names))

        for i, pred in enumerate(det):
            lw = max(round(sum(img_src.shape) / 2 * 0.002), 1)  # line width
            tf = max(lw - 1, 1)  # font thickness
            sf = lw / 4  # font scale
            pred[:, :4] = ops.scale_boxes(img.shape[2:], pred[:, :4], img_src.shape)
            results = pred.cpu().detach().numpy()
            for result in results:
                cls_name = self.names[int(result[5])]
                if cls_name == "notok":
                    self.fail += 1  # 增加notok标签的计数
                self.draw_box(img_src, result[:4], result[4], self.names[result[5]], lw, sf, tf)

        # Convert image back to BGR format for displaying with OpenCV
        #如果是系统，这句再加上
        #img_src_bgr = cv2.cvtColor(img_src, cv2.COLOR_RGB2BGR)

        return img_src


# if __name__ == '__main__':
#     #load a model
#     #model = YOLO(r'/root/yolov8/yolov8n.pt');
#     model = YOLO(r'../datasets/yolov8_base_weight.pt');
#     #model = YOLO(r'D:\pycharm\ultralytics-main-source\datasets\best.pt');
#     model.predict(
#         source=r'../datasets/30138508_2.jpg',
#         #source=r'D:\pycharm\ultralytics-main-source\datasets\1.png',
#         save=True,  #save predict results
#         imgsz = 640, # (int) size of input images as integer or w,h
#         conf = 0.25,  # object confidence threshold for detection (default 0.25 predict, 0.001 val)
#         iou = 0.5,   # intersection over union (IoU) threshold for NMS
#         show = False, # show results if possible
#         project = '../runs/predict',
#         #project =r'D:\pycharm\ultralytics-main-source\runs\predict',
#         name = 'exp_base', # (str, optional) experiment name, results saved to 'project/name' directory
#         save_txt = False, # save results as .txt file
#         save_conf = True, # save results with confidence scores
#         save_crop = False, # save cropped images with results
#         show_labels = True, # show object Labels in plots
#         show_conf = True, # show object confidence scores in plots
#         vid_stride = 1, # video frane-rote stride
#         line_width = 3, # bounding box thickness (pixels)
#         visualize = False, # visualize model features
#         augment = False, # apply image augmentation to prediction sources
#         agnostic_nms = False, # class-agnostic NMS
#         retina_masks = False,  # use high-resolution segmentation masks
#         boxes = True, # # Show boxes in segmentation predictions
#     )
if __name__ == '__main__':
    # 模型权重路径
    #weights_path = r'../datasets/yolov8n_fourhead.pt'  # 替换为您的权重文件路径
    #weights_path = (r'../datasets/yolov8n_epoch200_bs16_coslr.pt')  # 替换为您的权重文件路径
    weights_path = (r'/root/autodl-tmp/project/runs/train/open-enhance/exp_yolovn82/weights/best.pt')  # 替换为您的权重文件路径
    yolo = YOLOV8DetectionInfer(weights_path, 'cpu', 0.25, 0.5)
    # img_path = r'../datasets/6.jpg'
    # image = cv2.imread(img_path)
    # #cv2.imshow('Prediction Results', image)
    # image_result = yolo.infer(image)
    # cv2.imshow('Prediction Results', image_result)  # 显示带检测框的图像
    # #
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()
    # # 保存带检测框的图像到文件
    # output_path = '../runs/predict/exp/6_base.jpg'  # 设置输出路径
    # cv2.imwrite(output_path, image_result)  # 保存图像
    # print(f"Prediction result saved to {output_path}")
    '''
    文件夹一起检测
    '''

    # 输入图片文件夹路径
    input_folder = r'/root/autodl-tmp/project/split_datasets/PCB/test/images'  # 替换为您自己的文件夹路径
    output_folder = r'../runs/predict/exp_epoch200/'
    # input_folder = r'/root/autodl-tmp/PCB-open/train/images/'  # 替换为您自己的文件夹路径
    # output_folder = r'/root/autodl-tmp/PCB-open-predict/yolov8s/'  # 设置输出路径

    # 遍历文件夹中的所有图片文件
    for filename in os.listdir(input_folder):
        img_path = os.path.join(input_folder, filename)

        # 确保只处理图片文件
        if img_path.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff')):
            # 读取图片
            image = cv2.imread(img_path)

            # 进行推理
            image_result = yolo.infer(image)

            # 显示结果
            # cv2.imshow(f'Prediction Results - {filename}', image_result)
            # cv2.waitKey(0)

            # 保存带检测框的图像到文件
            output_path = os.path.join(output_folder, filename)
            cv2.imwrite(output_path, image_result)  # 保存图像
            print(f"Prediction result saved to {output_path}")

    cv2.destroyAllWindows()
