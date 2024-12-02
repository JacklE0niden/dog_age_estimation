import torch
from models.model import build_model
from datasets.eval_dataloader import get_eval_dataloader
import torch.nn as nn

def evaluate_model(data_dir, model_path):
    # 加载模型
    model = build_model(pretrained=False)
    model = torch.nn.DataParallel(model)
    model.load_state_dict(torch.load(model_path))
    model.eval()

    # 获取数据加载器
    dataloader = get_eval_dataloader(data_dir)

    # 计算 MSE
    criterion = nn.MSELoss()
    total_loss = 0.0
    total_samples = 0

    with torch.no_grad():
        for images, labels in dataloader:
            # 将数据转移到对应的设备
            images, labels = images.cuda(), labels.cuda()

            # 计算模型输出
            outputs = model(images)

            # 计算损失
            loss = criterion(outputs.squeeze(), labels.float())
            total_loss += loss.item() * len(labels)  # 累加损失并乘以batch大小
            total_samples += len(labels)

    # 计算平均损失
    avg_loss = total_loss / total_samples
    print(f"Evaluation complete. Average MSE loss: {avg_loss:.4f}")