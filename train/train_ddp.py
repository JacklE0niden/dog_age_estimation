import os
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler
import torch.optim as optim
import torch.nn as nn
# from datasets.dataloader import get_dataloader, DogAgeDataset
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from models.model import build_model
from datasets.dataloader import get_dataloader, DogAgeDataset
# from models.model_d import build_model

def train_model(data_dir, num_epochs=500, batch_size=32, learning_rate=0.001, rank=0):
    # 初始化模型，使用预训练权重
    model = build_model(pretrained=True).to(rank)  # 设置为True以使用预训练模型
    model = DDP(model, device_ids=[rank], output_device=rank, find_unused_parameters=True)

    # 加载训练集和验证集
    train_dataset = DogAgeDataset(
        img_dir=f'{data_dir}/trainset',
        annotations_file=f'{data_dir}/annotations/train.txt',
        transform=transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
    )
    train_sampler = DistributedSampler(dataset=train_dataset, shuffle=True)  # 使用分布式采样器
    train_loader = DataLoader(
        dataset=train_dataset,
        batch_size=batch_size,
        sampler=train_sampler,
        num_workers=4,  # 根据硬件调整
        pin_memory=True
    )

    # 定义损失函数和优化器
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-5)  # 添加L2正则化

    # 设置最小学习率
    min_lr = 1e-6  # 设置一个合理的最小学习率

    # 使用ReduceLROnPlateau调度器
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5, verbose=True, min_lr=min_lr)

    for epoch in range(num_epochs):
        model.train()
        train_sampler.set_epoch(epoch)  # 确保每个进程数据不重
        running_loss = 0.0

        for step, (images, labels) in enumerate(train_loader):
            images, labels = images.to(rank), labels.to(rank)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs.squeeze(), labels.float())
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

        avg_loss = running_loss / len(train_loader)
        print(f"Rank {rank}, Epoch [{epoch+1}/{num_epochs}], Average Loss: {avg_loss:.4f}")

        # 在每个epoch结束时更新学习率
        scheduler.step(avg_loss)  # 使用当前平均损失更新学习率

        # 打印当前学习率
        print(f'Epoch {epoch+1}, Learning Rate: {scheduler.optimizer.param_groups[0]["lr"]:.6f}')

    if rank == 0:  # 仅主进程保存模型
        torch.save(model.module.state_dict(), './saved_models/dog_age_ddp_efficientnet_with_resnet_fusion_norm_SE_epoch500_pretrained.pth')

def main():
    # 初始化分布式环境
    dist.init_process_group(backend="nccl")
    rank = dist.get_rank()  # 获取当前进程的 rank
    torch.cuda.set_device(rank)  # 将当前进程绑定到对应 GPU

    data_directory = './data'
    train_model(data_directory, rank=rank)

    dist.destroy_process_group()  # 销毁分布式环境

if __name__ == "__main__":
    main()