import os
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image

# 自定义数据集
# class DogAgeDataset(Dataset):
#     def __init__(self, img_dir, annotations_file, transform=None):
#         self.img_dir = img_dir
#         self.transform = transform
#         self.img_labels = []
        
#         # 读取标注文件
#         with open(annotations_file, 'r') as f:
#             for line in f:
#                 img_name, age = line.strip().split('\t')
#                 self.img_labels.append((img_name, int(age)))
    
#     def __len__(self):
#         return len(self.img_labels)
    
#     def __getitem__(self, idx):
#         img_path = os.path.join(self.img_dir, self.img_labels[idx][0])
#         image = Image.open(img_path).convert('RGB')
#         label = self.img_labels[idx][1]
        
#         if self.transform:
#             image = self.transform(image)
        
#         return image, label

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