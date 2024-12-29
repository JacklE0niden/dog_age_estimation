# TODO List

## 1. 调整 `train_ddp.py` 使其支持单卡和多卡训练
- 修改 `train_ddp.py` 代码，使其能在单卡和多卡训练环境中都能正确运行。
- 确保模型在多卡训练中使用 `DistributedDataParallel (DDP)`，并在单卡训练时能够正常运行。

## 2. 调整 `evaluate_ddp.py` 以支持加载单卡和多卡训练出来的权重
- 修改 `evaluate_ddp.py` 代码，使其能导入通过单卡或多卡训练得到的模型权重。
- 确保权重加载时能够兼容 DDP 包装，并能正确加载模型。

## 3. 修复预测文件名无法加载的问题
- 确保预测时文件名能够正确加载，并且与 `annotations` 文件中的标签对应。
- 修改预测时的数据处理方式，确保文件名与标签能够准确匹配。

## 4. 修复预测值没有个位数的问题
- 检查模型是否存在分类头，可能是因为分类头导致预测值不为个位数。
- 修改模型输出部分，确保预测值为正确的回归值。

## 5. 替换为更好的网络

SE100
[INFO] Validation complete. Average MSE Loss: 1433.0162
[INFO] Regression Accuracy: 1.33%


SE1000
[INFO] Validation complete. Average MSE Loss: 1457.6709
[INFO] Regression Accuracy: 1.60%


fusion_epoch500_unpretrained
[INFO] Validation complete. Average MSE Loss: 1378.0119
[INFO] Regression Accuracy: 2.27%

fusion_norm_SE_epoch500
[INFO] Validation complete. Average MSE Loss: 1377.2849
[INFO] Regression Accuracy: 2.27%