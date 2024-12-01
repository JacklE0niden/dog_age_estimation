import torch
import torch.distributed as dist
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from models.model import build_model
from datasets.dataloader import get_dataloader
import os

def setup_distributed():
    """初始化分布式环境"""
    print("[INFO] Initializing distributed environment...")
    dist.init_process_group(backend="nccl", init_method="env://")
    local_rank = int(os.getenv("LOCAL_RANK", 0))
    print(f"[INFO] Process {local_rank} initialized.")
    torch.cuda.set_device(local_rank)
    return local_rank

def cleanup_distributed():
    """清理分布式环境"""
    print("[INFO] Cleaning up distributed environment...")
    dist.destroy_process_group()

def train_model(local_rank, data_dir, num_epochs=10, batch_size=32, learning_rate=0.001):
    # 初始化分布式环境
    local_rank = setup_distributed()
    device = torch.device(f"cuda:{local_rank}")

    # 初始化模型
    print(f"[INFO] Process {local_rank}: Initializing model...")
    model = build_model(pretrained=True).to(device)
    model = nn.parallel.DistributedDataParallel(model, device_ids=[local_rank])
    print(f"[INFO] Process {local_rank}: Model initialized.")

    # 加载数据
    print(f"[INFO] Process {local_rank}: Loading data...")
    train_dataset = get_dataloader(img_dir=f'{data_dir}/trainset', 
                                    annotations_file=f'{data_dir}/annotations/train.txt')
    val_dataset = get_dataloader(img_dir=f'{data_dir}/valset', 
                                  annotations_file=f'{data_dir}/annotations/val.txt')

    train_sampler = DistributedSampler(train_dataset)
    val_sampler = DistributedSampler(val_dataset, shuffle=False)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, sampler=train_sampler)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, sampler=val_sampler)
    print(f"[INFO] Process {local_rank}: Data loaded.")

    # 定义损失函数和优化器
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    # 开始训练
    for epoch in range(num_epochs):
        print(f"[INFO] Process {local_rank}: Starting epoch {epoch+1}...")
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

            # 打印每 N 步损失
            if step % 10 == 0:
                print(f"[INFO] Process {local_rank}: Epoch [{epoch+1}/{num_epochs}], Step [{step}/{len(train_loader)}], Loss: {loss.item():.4f}")

        # 打印每个 epoch 的训练损失
        epoch_loss = running_loss / len(train_loader)
        if local_rank == 0:
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

        val_loss /= len(val_loader)
        if local_rank == 0:
            print(f"[RESULT] Epoch [{epoch+1}/{num_epochs}], Validation Loss: {val_loss:.4f}")

    # 保存模型（仅主进程保存）
    if local_rank == 0:
        os.makedirs('./saved_models', exist_ok=True)
        torch.save(model.state_dict(), './saved_models/dog_age_model.pth')
        print("[INFO] Training complete. Model saved.")

    # 清理分布式环境
    cleanup_distributed()

# 调用训练函数
if __name__ == "__main__":
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "12355"

    print("[INFO] Starting distributed training...")
    torch.multiprocessing.spawn(
        train_model,
        args=('./data',),
        nprocs=torch.cuda.device_count(),
        join=True
    )
    print("[INFO] Training completed.")