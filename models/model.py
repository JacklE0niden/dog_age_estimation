import torch.nn as nn
import torchvision.models as models
from efficientnet_pytorch import EfficientNet

def build_model(pretrained=True):
    model = models.resnet18(pretrained=pretrained)
    num_features = model.fc.in_features
    model.fc = nn.Linear(num_features, 1)  # 输出单个值用于回归
    return model


# def build_model(pretrained=True):
#     model = EfficientNet.from_pretrained('efficientnet-b0') if pretrained else EfficientNet.from_name('efficientnet-b0')
#     num_features = model._fc.in_features
#     model._fc = nn.Linear(num_features, 1)  # 输出单个值用于回归
#     return model