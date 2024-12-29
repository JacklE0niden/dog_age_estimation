import os
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import cv2
from PIL import Image
import numpy as np

# 自定义函数检测并裁剪宠物的脸部
from mtcnn import MTCNN
# import cv2

def detect_and_crop_face(image):
    detector = MTCNN()
    # 检测人脸（宠物脸也可能有效）
    faces = detector.detect_faces(image)
    
    if len(faces) > 0:
        x, y, w, h = faces[0]['box']
        return image[y:y+h, x:x+w]  # 裁剪脸部区域
    return image  # 如果没有检测到脸部，返回原图

class DogAgeDataset(Dataset):
    def __init__(self, img_dir, annotations_file, transform=None):
        self.img_dir = img_dir
        self.annotations_file = annotations_file
        self.transform = transform
        
        # 读取注释文件，假设每行包含图像文件名和对应的标签
        with open(annotations_file, 'r') as f:
            self.annotations = f.readlines()

    def __len__(self):
        return len(self.annotations)

    def __getitem__(self, idx):
        # 获取当前索引的文件名和标签
        img_name, label = self.annotations[idx].strip().split('\t')
        img_path = os.path.join(self.img_dir, img_name)
        image = Image.open(img_path).convert('RGB')
        # print(img_path)
        label = float(label)  # 确保标签是浮动类型
        
        if self.transform:
            image = self.transform(image)
        
        return image, label
# DataLoader 获取函数
def get_dataloader(img_dir, annotations_file, batch_size=32, train=True):
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip() if train else transforms.Lambda(lambda x: x),  # 替换 Identity
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    dataset = DogAgeDataset(img_dir=img_dir, annotations_file=annotations_file, transform=transform)
    return DataLoader(dataset, batch_size=batch_size, shuffle=train)