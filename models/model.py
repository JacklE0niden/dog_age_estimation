import torch
import torch.nn as nn
import torchvision.models as models
from efficientnet_pytorch import EfficientNet
from .se import SEBlock  # 导入SE模块
import cv2  # 导入OpenCV

# 定义一个函数来检测和裁剪宠物的脸部
def detect_and_crop_face(image):
    # 加载Haar级联分类器
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalcatface.xml')
    
    # 将图像转换为灰度图
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # 检测脸部
    faces = face_cascade.detectMultiScale(gray_image, scaleFactor=1.1, minNeighbors=5)
    
    # 如果检测到脸部，裁剪并返回第一个检测到的脸部
    if len(faces) > 0:
        (x, y, w, h) = faces[0]
        return image[y:y+h, x:x+w]  # 裁剪脸部区域
    return image  # 如果没有检测到脸部，返回原图

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
                nn.Dropout(0.3)
            ])
            prev_dim = dim
        
        # 最后的回归层
        layers.append(nn.Linear(prev_dim, 1))
        
        self.head = nn.Sequential(*layers)
    
    def forward(self, x):
        return self.head(x)

class DogAgeEstimator(nn.Module):
    def __init__(self, use_pretrained=False):
        super().__init__()
        
        # 多个backbone
        self.resnet = models.resnet34(pretrained=use_pretrained)
        self.efficient_net = EfficientNet.from_pretrained('efficientnet-b0') if use_pretrained \
            else EfficientNet.from_name('efficientnet-b0')
        
        # 获取特征维度
        resnet_features = self.resnet.fc.in_features
        efficientnet_features = self.efficient_net._fc.in_features
        
        # 移除原始的分类层
        self.resnet = nn.Sequential(*list(self.resnet.children())[:-1])
        self.efficient_net.set_swish(memory_efficient=False)
        
        # 特征融合层
        total_features = resnet_features + efficientnet_features
        self.fusion = nn.Sequential(
            nn.Linear(total_features, total_features // 2),
            nn.BatchNorm1d(total_features // 2),
            nn.ReLU(inplace=True)
        )
        
        # 添加SE块
        self.se_block = SEBlock(total_features // 2)  # 使用已定义的SEBlock
        
        # 回归头部
        self.regression_head = AgeRegressionHead(
            in_features=total_features // 2,
            hidden_dims=[512, 256]
        )
        
        # 添加整数约束
        self.round = lambda x: torch.round(x)
    
    def forward(self, x):
        # ResNet特征
        resnet_feat = self.resnet(x)
        resnet_feat = torch.flatten(resnet_feat, 1)
        
        # EfficientNet特征
        efficientnet_feat = self.efficient_net.extract_features(x)
        efficientnet_feat = self.efficient_net._avg_pooling(efficientnet_feat)
        efficientnet_feat = torch.flatten(efficientnet_feat, 1)
        
        # 特征融合
        combined_feat = torch.cat([resnet_feat, efficientnet_feat], dim=1)
        fused_feat = self.fusion(combined_feat)
        
        # 调整形状以适应SEBlock
        fused_feat = fused_feat.unsqueeze(2).unsqueeze(3)  # 添加两个维度
        
        # 通过SE块
        fused_feat = self.se_block(fused_feat)  # 调用SEBlock
        
        # 将SE块的输出展平
        fused_feat = fused_feat.view(fused_feat.size(0), -1)  # 展平为 [batch_size, features]
        
        # 回归预测
        age_pred = self.regression_head(fused_feat)
        
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
    
    return DogAgeEstimator(use_pretrained=pretrained)