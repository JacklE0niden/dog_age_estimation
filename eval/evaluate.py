import torch
from models.model import build_model
from datasets.data_loader import get_dataloader

def evaluate_model(data_dir, model_path):
    model = build_model(pretrained=False)
    model.load_state_dict(torch.load(model_path))
    model.eval()
    dataloader = get_dataloader(data_dir, train=False)

    with torch.no_grad():
        total_loss = 0.0
        for images, labels in dataloader:
            outputs = model(images)
            # 计算MSE或其他评估指标
            pass

    print("Evaluation complete.")