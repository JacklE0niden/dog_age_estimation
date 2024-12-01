import os
import argparse
import torch
import torch.distributed as dist
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from models.model import build_model
from datasets.dataloader import get_dataloader

# 设置 argparse 参数解析
parser = argparse.ArgumentParser(description='Distributed training for dog age estimation.')
parser.add_argument('--local_rank', type=int, default=0, help="Local rank for distributed training.")
parser.add_argument('--data_dir', type=str, default='./data', help="Path to the dataset directory.")
parser.add_argument('--num_epochs', type=int, default=10, help="Number of training epochs.")
parser.add_argument('--batch_size', type=int, default=32, help="Batch size for training.")
parser.add_argument('--learning_rate', type=float, default=0.001, help="Learning rate for optimizer.")
opt = parser.parse_args()

def setup_distributed(local_rank):
    """初始化分布式环境"""
    dist.init_process_group(backend="nccl", init_method="env://")
    torch.cuda.set_device(local_rank)
    print(f"[INFO] Process {local_rank} initialized.")
    return local_rank

def cleanup_distributed():
    """清理分布式环境"""
    dist.destroy_process_group()

def train_model(local_rank, data_dir, num_epochs, batch_size, learning_rate):
    """训练模型的主函数"""
    # 初始化分布式环境
    local_rank = setup_distributed(local_rank)
    device = torch.device(f"cuda:{local_rank}")

    # 初始化模型
    model = build_model(pretrained=True).to(device)
    model = nn.parallel.DistributedDataParallel(model, device_ids=[local_rank])

    # 加载数据
    train_dataset = get_dataloader(img_dir=f'{data_dir}/trainset', 
                                    annotations_file=f'{data_dir}/annotations/train.txt')
    val_dataset = get_dataloader(img_dir=f'{data_dir}/valset', 
                                  annotations_file=f'{data_dir}/annotations/val.txt')

    train_sampler = DistributedSampler(train_dataset)
    val_sampler = DistributedSampler(val_dataset, shuffle=False)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, sampler=train_sampler)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, sampler=val_sampler)

    # 定义损失函数和优化器
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    # 训练循环
    for epoch in range(num_epochs):
        train_sampler.set_epoch(epoch)
        model.train()
        running_loss = 0.0

        for step, (images, labels) in enumerate(train_loader):
            images, labels = images.to(device), labels.to(device)

            # 前向传播
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs.squeeze(), labels.float())

            # 反向传播和优化
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

            if step % 10 == 0 and local_rank == 0:
                print(f"[INFO] Epoch [{epoch+1}/{num_epochs}], Step [{step}/{len(train_loader)}], Loss: {loss.item():.4f}")

        if local_rank == 0:
            epoch_loss = running_loss / len(train_loader)
            print(f"[RESULT] Epoch [{epoch+1}/{num_epochs}], Training Loss: {epoch_loss:.4f}")

        # 验证模型
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                loss = criterion(outputs.squeeze(), labels.float())
                val_loss += loss.item()

        if local_rank == 0:
            val_loss /= len(val_loader)
            print(f"[RESULT] Epoch [{epoch+1}/{num_epochs}], Validation Loss: {val_loss:.4f}")

    if local_rank == 0:
        os.makedirs('./saved_models', exist_ok=True)
        torch.save(model.state_dict(), './saved_models/dog_age_model.pth')
        print("[INFO] Training complete. Model saved.")

    cleanup_distributed()

if __name__ == "__main__":
    # 从 argparse 获取参数
    local_rank = opt.local_rank
    data_dir = opt.data_dir
    num_epochs = opt.num_epochs
    batch_size = opt.batch_size
    learning_rate = opt.learning_rate

    # 初始化分布式训练
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "12355"

    torch.distributed.init_process_group(backend="nccl", init_method="env://")
    if torch.distributed.get_world_size() != torch.cuda.device_count():
        raise ValueError(f"[ERROR] Available GPUs ({torch.cuda.device_count()}) do not match world size ({torch.distributed.get_world_size()})!")

    # 启动训练
    torch.multiprocessing.spawn(
        train_model,
        args=(data_dir, num_epochs, batch_size, learning_rate),
        nprocs=torch.cuda.device_count(),
        join=True
    )