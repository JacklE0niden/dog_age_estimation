import torch
import torch.nn as nn
import torchvision.models as models
from .se import SEBlock  # 导入SE模块

class AgeRegressionHead(nn.Module):
    def __init__(self, in_features, hidden_dims=[512, 256]):
        super().__init__()
        layers = []
        prev_dim = in_features
        
        # 创建多层MLP
        for dim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, dim),
                nn.BatchNorm1d(dim),
                nn.ReLU(inplace=True),
                nn.Dropout(0.5)  # 增加Dropout层，防止过拟合
            ])
            prev_dim = dim
        
        # 最后的回归层
        layers.append(nn.Linear(prev_dim, 1))
        
        self.head = nn.Sequential(*layers)
    
    def forward(self, x):
        return self.head(x)

class ResNetWithSE(nn.Module):
    def __init__(self, use_pretrained=False):
        super().__init__()
        
        # 使用ResNet作为特征提取器
        self.resnet = models.resnet34(pretrained=use_pretrained)
        
        # 获取特征维度
        resnet_features = self.resnet.fc.in_features
        
        # 移除原始的分类层
        self.resnet = nn.Sequential(*list(self.resnet.children())[:-1])
        
        # 在每个卷积块后添加SEBlock
        self.se_blocks = nn.ModuleList([
            SEBlock(in_channels) for in_channels in [64, 128, 256, 512]
        ])
        
        # 回归头部
        self.regression_head = AgeRegressionHead(
            in_features=resnet_features,
            hidden_dims=[512, 256]
        )
        
        # 添加整数约束
        self.round = lambda x: torch.round(x)
    
    def forward(self, x):
        # 通过ResNet
        for i, layer in enumerate(self.resnet):
            x = layer(x)
            if i in [4, 5, 6, 7]:  # 在特定层后添加SEBlock
                x = self.se_blocks[i - 4](x)  # 对应的SEBlock

        # Flatten
        resnet_feat = torch.flatten(x, 1)
        
        # 回归预测
        age_pred = self.regression_head(resnet_feat)
        
        # 训练时返回原始值，推理时返回整数
        if self.training:
            return age_pred
        else:
            return self.round(age_pred)

def build_model(pretrained=False):
    # 打印是否使用预训练模型
    if pretrained:
        print("Using pretrained model.")
    else:
        print("Not using pretrained model.")
    
    return ResNetWithSE(use_pretrained=pretrained)