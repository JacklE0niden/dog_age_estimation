import os

# 定义文件路径
pred_file_path = '/mnt/pami26/zengyi/dlearning/dog_age_estimation/use.txt'
annotation_file_path = '/mnt/pami26/zengyi/dlearning/dog_age_estimation/data/annotations/val.txt'

# 读取预测文件和注释文件
with open(pred_file_path, 'r') as pred_file:
    pred_lines = pred_file.readlines()

with open(annotation_file_path, 'r') as annotation_file:
    annotation_lines = annotation_file.readlines()

# 确保注释文件的行数与预测文件相同
if len(pred_lines) != len(annotation_lines):
    print("预测文件和注释文件的行数不一致，请检查文件。")
else:
    # 更新预测文件中的条目
    updated_lines = []
    for pred_line, annotation_line in zip(pred_lines, annotation_lines):
        # 提取预测文件中的文件名（去掉年龄部分）
        pred_parts = pred_line.strip().split()  # 假设用制表符分隔
        
        # 检查分割后的列表长度
        # if len(pred_parts) < 3:
        #     print(f"警告: 行格式不正确，跳过该行: {pred_line.strip()}")
        #     updated_lines.append(pred_line)  # 保留原行
        #     continue
        
        # 使用注释文件中的文件名替换预测文件中的文件名
        annotation_file_name = annotation_line.strip().split()  # 注释文件中的文件名
        print("annotation:", annotation_file_name, "pred_parts:", pred_parts)
        updated_line = f"{annotation_file_name[0]}\t{pred_parts[1]}\n"
        updated_lines.append(updated_line)

    # 将更新后的内容写回预测文件
    with open(pred_file_path, 'w') as pred_file:
        pred_file.writelines(updated_lines)

    print("文件名更新完成。")