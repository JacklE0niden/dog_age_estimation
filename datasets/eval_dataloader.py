# datasets/eval_dataloader.py
import torch
from torch.utils.data import DataLoader
from torchvision import transforms
from datasets.dataloader import DogAgeDataset

def get_eval_dataloader(data_dir, batch_size=32):
    # 设置数据预处理
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    # 创建测试集数据集
    val_dataset = DogAgeDataset(
        img_dir=f'{data_dir}/valset',
        annotations_file=f'{data_dir}/annotations/val.txt',
        transform=transform
    )

    # 创建数据加载器
    val_loader = DataLoader(
        dataset=val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=True
    )
    
    return val_loader