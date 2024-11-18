import torch
from dataloader import get_dataloader
import matplotlib.pyplot as plt
import os

if __name__ == "__main__":
    # 加载 DataLoader
    test_loader = get_dataloader(img_dir='./data/trainset', annotations_file='./data/annotations/train.txt', batch_size=4)

    # 获取一个批次的数据
    images, labels = next(iter(test_loader))

    # 检查 DataLoader 是否正常运行
    print(f"Batch size: {images.size()}")  # 输出批次的大小
    print(f"Labels: {labels}")  # 输出对应的标签

    # 创建 demo 文件夹，如果不存在
    os.makedirs('demo', exist_ok=True)

    # 显示并保存每张图像
    fig, axes = plt.subplots(1, 4, figsize=(15, 5))
    for i, (img, label) in enumerate(zip(images, labels)):
        img = img.permute(1, 2, 0)  # 将图像从 (C, H, W) 转换为 (H, W, C)
        img = img * torch.tensor([0.229, 0.224, 0.225]) + torch.tensor([0.485, 0.456, 0.406])  # 反标准化
        img = torch.clip(img, 0, 1)  # 将像素值限制在 [0, 1] 范围内

        axes[i].imshow(img)
        axes[i].set_title(f"Age: {label}")
        axes[i].axis('off')

        # 保存单独的子图
        fig_single, ax_single = plt.subplots()
        ax_single.imshow(img)
        ax_single.set_title(f"Age: {label}")
        ax_single.axis('off')
        fig_single.savefig(f'demo/image_{i}.png')
        plt.close(fig_single)  # 关闭单独保存的图像以释放内存

    plt.show()  # 显示整体图像

    plt.tight_layout()
    plt.close(fig)
