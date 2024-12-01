import torch
import torch.optim as optim
import torch.nn as nn
from datasets.dataloader import get_dataloader
from models.model import build_model

def train_model(data_dir, num_epochs=10, batch_size=32, learning_rate=0.001):
    # 初始化模型
    model = build_model(pretrained=True)
    print("CUDA Available? ", torch.cuda.is_available())
    model = model.to(torch.device("cuda" if torch.cuda.is_available() else "cpu"))

    # 加载训练集和验证集
    train_loader = get_dataloader(img_dir=f'{data_dir}/trainset', annotations_file=f'{data_dir}/annotations/train.txt', batch_size=batch_size)
    val_loader = get_dataloader(img_dir=f'{data_dir}/valset', annotations_file=f'{data_dir}/annotations/val.txt', batch_size=batch_size)
    print(f"训练集大小: {len(train_loader)}")
    print(f"验证集大小: {len(val_loader)}")
    
    # 定义损失函数和优化器
    criterion = nn.MSELoss()  # 用于回归问题
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    
    for epoch in range(num_epochs):
        print("Epoch", epoch + 1)
        model.train()
        running_loss = 0.0

        for step, (images, labels) in enumerate(train_loader):
            print(f"Processing batch {step + 1}/{len(train_loader)}")
            
            # 将数据移到设备上（GPU 或 CPU）
            images, labels = images.to(torch.device("cuda" if torch.cuda.is_available() else "cpu")), labels.to(torch.device("cuda" if torch.cuda.is_available() else "cpu"))

            # 前向传播
            optimizer.zero_grad()
            outputs = model(images)
            
            # 计算损失
            loss = criterion(outputs.squeeze(), labels.float())

            # 反向传播和优化
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

            # 打印每个 batch 的损失
            print(f"Batch {step + 1}/{len(train_loader)}, Loss: {loss.item():.4f}")
            
        # 打印每个 epoch 的损失
        epoch_loss = running_loss / len(train_loader)
        print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {epoch_loss:.4f}")
        # 验证模型
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(torch.device("cuda" if torch.cuda.is_available() else "cpu")), labels.to(torch.device("cuda" if torch.cuda.is_available() else "cpu"))
                outputs = model(images)
                loss = criterion(outputs.squeeze(), labels.float())
                val_loss += loss.item()
        
        val_loss /= len(val_loader)
        print(f'Validation Loss: {val_loss:.4f}')

    # 保存训练好的模型
    torch.save(model.state_dict(), './saved_models/dog_age_model.pth')
    print("Training complete. Model saved.")

# 调用训练函数
if __name__ == "__main__":
    data_directory = './data'
    train_model(data_directory)