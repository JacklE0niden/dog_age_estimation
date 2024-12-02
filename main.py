from train.train import train_model
from eval.evaluate_ddp import evaluate_model, evaluate
import argparse

def main():
    # 创建 ArgumentParser
    parser = argparse.ArgumentParser(description="Train or Evaluate Model")
    parser.add_argument('--data_dir', type=str, required=True, help='Path to the data directory')
    parser.add_argument('--model_path', type=str, required=False, help='Path to the model checkpoint (required for evaluation)')
    parser.add_argument('--phase', type=str, choices=['train', 'evaluate'], required=True, help="Specify whether to train or evaluate the model")
    parser.add_argument('--num_epochs', type=int, default=20, help="Number of epochs for training (only for train phase)")
    parser.add_argument('--world_size', type=int, default=1, help="Number of GPUs for distributed mode (optional)")

    args = parser.parse_args()

    # 根据 phase 执行对应逻辑
    if args.phase == 'train':
        train_model(data_dir=args.data_dir, num_epochs=args.num_epochs)
    elif args.phase == 'evaluate':
        if not args.model_path:
            raise ValueError("Model path must be specified for evaluation.")
        evaluate_model(data_dir=args.data_dir, model_path=args.model_path)

if __name__ == "__main__":
    main()