from train.train import train_model
from eval.evaluate import evaluate_model

if __name__ == "__main__":
    train_model(data_dir='data/train', num_epochs=20)
    evaluate_model(data_dir='data/val', model_path='dog_age_model.pth')