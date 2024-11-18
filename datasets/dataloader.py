import os
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image

# DogAgeDataset 和 get_dataloader 定义保持不变

class DogAgeDataset(Dataset):
    def __init__(self, img_dir, annotations_file, transform=None):
        self.img_dir = img_dir
        self.transform = transform
        self.img_labels = []
        
        # 读取标注文件
        with open(annotations_file, 'r') as f:
            for line in f:
                img_name, age = line.strip().split('\t')
                self.img_labels.append((img_name, int(age)))
    
    def __len__(self):
        return len(self.img_labels)
    
    def __getitem__(self, idx):
        img_path = os.path.join(self.img_dir, self.img_labels[idx][0])
        image = Image.open(img_path).convert('RGB')
        label = self.img_labels[idx][1]
        
        if self.transform:
            image = self.transform(image)
        
        return image, label

def get_dataloader(img_dir, annotations_file, batch_size=32, train=True):
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip() if train else transforms.Identity(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    dataset = DogAgeDataset(img_dir=img_dir, annotations_file=annotations_file, transform=transform)
    return DataLoader(dataset, batch_size=batch_size, shuffle=train)