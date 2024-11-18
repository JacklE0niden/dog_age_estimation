import torch.nn as nn
import torchvision.models as models

def build_model(pretrained=True):
    model = models.resnet18(pretrained=pretrained)
    num_features = model.fc.in_features
    model.fc = nn.Linear(num_features, 1)  # 输出单个值用于回归
    return model