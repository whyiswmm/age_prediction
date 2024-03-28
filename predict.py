from facenet_pytorch import MTCNN
# from AgeNet.models import Model
import torch
from torchvision import transforms as T
from PIL import Image
import numpy as np
import cv2
import matplotlib.pyplot as plt
import os
import argparse
import re
from transformers import ViTForImageClassification, ViTImageProcessor, AutoModelForImageClassification
from peft import PeftConfig, PeftModel, LoraConfig, get_peft_model
# from transformers import AutoModelForImageClassification, TrainingArguments, Trainer, AutoImageProcessor
import torch.nn as nn


class AgeEstimator():
    def __init__(self, face_size = 224, weights = None, device = 'cpu', tpx = 500):  
        
        self.thickness_per_pixels = tpx
        
        if isinstance(face_size, int):
            self.face_size = (face_size, face_size)
        else:
            self.face_size = face_size   
        self.device = 'cuda' # 默认有cuda，不cuda就直接报错吧
        repo_name = 'lora_model_100_epoch_10fenlei_0325'
        base_model_name_or_path = 'nateraw/vit-age-classifier'
        model = AutoModelForImageClassification.from_pretrained(
            base_model_name_or_path,
            ignore_mismatched_sizes=True,  # provide this in case you're planning to fine-tune an already fine-tuned checkpoint
        )
        model.classifier = nn.Linear(in_features=768, out_features=10, bias=True)
        # load the Lora model
        self.model = PeftModel.from_pretrained(model, repo_name).to(self.device)
        # load face detection model
        self.facenet_model = MTCNN(device = self.device)
        self.transforms = ViTImageProcessor.from_pretrained('nateraw/vit-age-classifier')
        self.all_face = 0 # 检测到的所有的人脸数
        self.underage_predict = 0 # 预测出的未成年人脸数
        # self.model.eval()
        self.age_list = []   # 所有人脸的具体年龄段放在这里
        self.label_list = [] # 所有人脸的年龄段标签放在这里
        self.filename = '' # 定义一个空名，以便写入
        self.num_img = 0  # 有人脸的图片数目
        self.num_underage = 0 # 出现未成年人脸的图片数        

    @staticmethod
    def plot_box_and_label(image, lw, box, label='', color=(128, 128, 128), txt_color=(255, 255, 255)):
        # Add one xyxy box to image with label
        p1, p2 = (int(box[0]), int(box[1])), (int(box[2]), int(box[3]))
        cv2.rectangle(image, p1, p2, color, thickness=lw, lineType=cv2.LINE_AA)
        if label:
            tf = max(lw - 1, 1)  # font thickness
            w, h = cv2.getTextSize(label, 0, fontScale=lw / 3, thickness=tf)[0]  # text width, height
            outside = p1[1] - h - 3 >= 0  # label fits outside box
            p2 = p1[0] + w, p1[1] - h - 3 if outside else p1[1] + h + 3
            cv2.rectangle(image, p1, p2, color, -1, cv2.LINE_AA)  # filled
            cv2.putText(image, label, (p1[0], p1[1] - 2 if outside else p1[1] + h + 2), 0, lw / 3, txt_color,
                        thickness=tf, lineType=cv2.LINE_AA)
    
    def padding_face(self, box, padding = 10):
        return [
            box[0] - padding,
            box[1] - padding,
            box[2] + padding,
            box[3] + padding
        ]
            
    def predict(self, img_path, min_prob = 0.9):
        image = img_path
        ndarray_image = np.array(img_path)
        image_shape = ndarray_image.shape
        bboxes, prob = self.facenet_model.detect(image)
        if prob[0] == None:
            return ndarray_image, 0
        bboxes = bboxes[prob > min_prob]
        if len(bboxes)==0:
            return ndarray_image, 0

        face_images = []
        self.num_img += 1

        for box in bboxes:
            box = np.clip(box, 0, np.inf).astype(np.uint32)
            padding = max(image_shape) * 5 / self.thickness_per_pixels
            padding = int(max(padding, 10))
            box = self.padding_face(box, padding)
            
            face = image.crop(box)
            transformed_face = self.transforms(face, return_tensors='pt')
            data = transformed_face['pixel_values'].squeeze(0)
            face_images.append(data)
        
        stack_face_images = torch.stack(face_images, dim = 0).to(self.device)

        output = self.model(stack_face_images)
        proba = output.logits.softmax(1)
        preds = proba.argmax(1) # Predicted Classes

        a = {'0': "0-2", '1': "3-9", '2':  "10-14", '3': "15-18", '4': "19-29", '5': "30-39", '6': "40-49", '7': "50-59", '8': "60-69", '9': "more than 70"}
        # a = {'0': "0-2", '1': "3-9", '2':  "10-19", '3': "20-29", '4': "30-39", '5': "40-49", '6': "50-59", '7': "60-69", '8': "more than 70"}
        underage_show = 0
        for i, box in enumerate(bboxes): 
            box = np.clip(box, 0, np.inf).astype(np.uint32)
            
            thickness = max(image_shape)/400
            
            thickness = int(max(np.ceil(thickness), 1))
            number = re.findall(r'\d+', str(preds[i]))
            ages = a[number[0]]
            self.age_list.append(ages)
            label = 'adult' if preds[i].item()>3 else "underage"
            self.label_list.append(label)
            self.all_face += 1
            if preds[i].item()<=3:
                self.underage_predict += 1
                underage_show += 1  #记录是否有未成年人
            label += f" {ages}"
            self.plot_box_and_label(ndarray_image, thickness, box, label, color = (255, 0, 0))
        if underage_show:
            self.num_underage += 1
        return ndarray_image, underage_show
        
    
    def count_zero(self, filename):
        self.all_face = 0
        self.underage_predict = 0
        self.age_list = []   
        self.label_list = []
        self.filename = filename
        self.num_img = 0
        self.num_underage = 0

    def get_record(self):
        record_age = ','.join(self.age_list)
        # record_label = ' '.join(self.label_list)
        if self.all_face == 0:
            underage_rate = 0
        else:
            underage_rate = round(self.underage_predict / self.all_face * 100, 2)

        if self.num_img == 0:
            underage_rate_1 = 0
        else:
            underage_rate_1 = round(self.num_underage / self.num_img * 100, 2)
        underage_all = f'{self.underage_predict}//{self.all_face}'
        underage_img = f'{self.num_underage}//{self.num_img}'
        # record = f'{self.filename} [{record_age}] {underage_all} face: {underage_rate}% img: {underage_rate_1} \n'
        record = f'{self.filename} {underage_all} face: {underage_rate}% {underage_img} img: {underage_rate_1}% \n'

        return record
