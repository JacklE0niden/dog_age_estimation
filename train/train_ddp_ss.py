import os
import torch
import torch.distributed as dist
import optuna
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP
from torchvision import transforms
from models.model import build_model
from datasets.dataloader import get_dataloader, DogAgeDataset
# 假设 build_model 和 DogAgeDataset 已在其他地方定义
# from your_model_file import build_model
# from your_dataset_file import DogAgeDataset

def objective(trial, rank, data_dir):
    # 从 Optuna 中获取超参数
    learning_rate = trial.suggest_loguniform('learning_rate', 1e-5, 1e-1)
    batch_size = trial.suggest_int('batch_size', 16, 64)
    # num_epochs = trial.suggest_int('num_epochs', 100, 500)
    num_epochs = 1  # 例如，设置为200个epoch  
    # 初始化模型，使用预训练权重
    model = build_model(pretrained=True).to(rank)
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
    train_sampler = DistributedSampler(dataset=train_dataset, shuffle=True)
    train_loader = DataLoader(
        dataset=train_dataset,
        batch_size=batch_size,
        sampler=train_sampler,
        num_workers=4,
        pin_memory=True
    )

    # 定义损失函数和优化器
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    # 使用StepLR调度器
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)

    running_loss = 0.0
    for epoch in range(num_epochs):
        model.train()
        train_sampler.set_epoch(epoch)
        
        for step, (images, labels) in enumerate(train_loader):
            images, labels = images.to(rank), labels.to(rank)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs.squeeze(), labels.float())
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

        print(f"Rank {rank}, Epoch [{epoch+1}/{num_epochs}], Average Loss: {running_loss / len(train_loader):.4f}")

        # 在每个epoch结束时更新学习率
        scheduler.step()
        print(f'Epoch {epoch+1}, Learning Rate: {scheduler.get_last_lr()[0]}')

    if rank == 0:  # 仅主进程保存模型
        torch.save(model.module.state_dict(), './saved_models/og_age_model_ddp_efficientnet_with_resnet_fusion_norm_SE_epoch500_pretrained.pth')

def train_model(data_dir, rank=0):
    # 使用Optuna进行超参数优化
    study = optuna.create_study(direction='minimize')
    study.optimize(lambda trial: objective(trial, rank, data_dir), n_trials=10)  # 进行10次试验

    # 输出最佳超参数
    print("Best hyperparameters: ", study.best_params)

    # 继续使用最佳超参数进行最终训练
    best_params = study.best_params
    # 使用最佳超参数进行最终训练的代码
    # 这里可以添加最终训练的逻辑

def main():
    # 初始化分布式环境
    dist.init_process_group(backend="nccl")
    rank = dist.get_rank()  # 获取当前进程的 rank
    torch.cuda.set_device(rank)  # 将当前进程绑定到对应 GPU

    data_directory = './data'  # 替换为实际数据目录
    train_model(data_directory, rank=rank)

    dist.destroy_process_group()  # 销毁分布式环境

if __name__ == "__main__":
    main()