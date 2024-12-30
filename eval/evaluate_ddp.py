import os
import torch
import torch.nn as nn
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
from torchvision import transforms
from datasets.dataloader import DogAgeDataset
from models.model import build_model
from torch.utils.data.distributed import DistributedSampler

# 修改后的代码
def evaluate_model(data_dir, model_path, rank=0, batch_size=16, tolerance=1):
    # 初始化模型
    model = build_model(pretrained=False).to(rank)

    # 加载模型权重
    print(f"[INFO] Loading model weights from {model_path}...")
    checkpoint = torch.load(model_path, map_location=f"cuda:{rank}")
    model.load_state_dict(checkpoint)
    print("[INFO] Model weights loaded successfully.")

    # 将模型传递给 DDP
    model = DDP(model, device_ids=[rank], output_device=rank)

    # 加载验证数据集
    val_dataset = DogAgeDataset(
        img_dir=f'{data_dir}/valset',
        annotations_file=f'{data_dir}/annotations/val.txt',
        transform=transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
    )

    # 使用 DistributedSampler 来确保每个进程只处理部分数据
    val_sampler = DistributedSampler(val_dataset, num_replicas=dist.get_world_size(), rank=rank, shuffle=False)

    val_loader = DataLoader(
        dataset=val_dataset,
        batch_size=batch_size,
        shuffle=False,  # shuffle由Sampler控制
        num_workers=4,
        pin_memory=True,
        sampler=val_sampler
    )

    # 定义损失函数
    criterion = nn.MSELoss()

    # 保存预测结果到文件
    pred_filename = "SE_pred_dog_age_model_ddp_resnet_SE_detection_epoch1000_pretrained.txt"
    # /mnt/pami26/zengyi/dlearning/dog_age_estimation/saved_models/dog_age_model_ddp_resnet_SE_detection_epoch1000_pretrained.pth
    total_loss = 0.0
    total_samples = 0
    correct_predictions = 0

    with open(pred_filename, 'a') as pred_file:
        with torch.no_grad():
            for step, (images, labels) in enumerate(val_loader):  # 只解包images和labels
                images, labels = images.to(rank), labels.to(rank)

                # 前向传播
                outputs = model(images)
                loss = criterion(outputs.squeeze(), labels.float())

                # 累加损失
                total_loss += loss.item() * labels.size(0)
                total_samples += labels.size(0)

                # 计算回归正确率
                predicted_labels = outputs.squeeze().cpu().numpy()
                actual_labels = labels.cpu().numpy()

                # 计算误差小于 tolerance 的样本数
                correct_predictions += sum(abs(predicted_labels - actual_labels) <= tolerance)

                # 将预测结果写入文件，包含文件名
                for i in range(len(images)):
                    # 获取当前图像的文件名
                    img_filename = val_dataset.annotations[i].strip().split('\t')[0]  # 从annotations中获取文件名
                    pred_file.write(f"{img_filename}\t{int(round(predicted_labels[i]))}\n")

                if rank == 0:  # 仅主进程打印损失
                    print(f"Rank {rank}, Batch [{step+1}/{len(val_loader)}], Loss: {loss.item():.4f}")

    # 计算平均损失和回归正确率
    avg_loss = total_loss / total_samples
    regression_accuracy = correct_predictions / total_samples * 100

    if rank == 0:  # 仅主进程打印最终结果
        print(f"[INFO] Validation complete. Average MSE Loss: {avg_loss:.4f}")
        print(f"[INFO] Regression Accuracy: {regression_accuracy:.2f}%")

    return avg_loss, regression_accuracy

def main():
    # 初始化分布式环境
    dist.init_process_group(backend="nccl")
    rank = dist.get_rank()  # 获取当前进程的 rank
    torch.cuda.set_device(rank)  # 将当前进程绑定到对应 GPU

    # 数据目录和模型路径
    data_directory = './data'
    model_path = './saved_models/dog_age_model_ddp_resnet_SE_detection_epoch1000_pretrained.pth'
    # /mnt/pami26/zengyi/dlearning/dog_age_estimation/saved_models/dog_age_model_ddp_resnet_SE_detection_epoch1000_pretrained.pth
    # /mnt/pami26/zengyi/dlearning/dog_age_estimation/saved_models/dog_age_model_ddp_efficientnet_with_resnet_fusion_epoch500_unpretrained.pth
    # /mnt/pami26/zengyi/dlearning/dog_age_estimation/saved_models/dog_age_model_ddp_resnet_SE_detection_epoch1000_pretrained.pth
    # /mnt/pami26/zengyi/dlearning/dog_age_estimation/saved_models/dog_age_model_ddp_efficientnet_with_resnet_fusion_epoch500_unpretrained.pth
    # /mnt/pami26/zengyi/dlearning/dog_age_estimation/saved_models/dog_age_model_ddp_resnet_SE_detection_epoch100_pretrained.pth
    # /mnt/pami26/zengyi/dlearning/dog_age_estimation/saved_models/dog_age_ddp_efficientnet_with_resnet_fusion_norm_SE_epoch500_pretrained.pth
    batch_size = 32

    evaluate_model(data_dir=data_directory, model_path=model_path, rank=rank, batch_size=batch_size)

    # 销毁分布式环境
    dist.destroy_process_group()


if __name__ == "__main__":
    main()