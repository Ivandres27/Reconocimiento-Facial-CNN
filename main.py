import torch
from data import get_dataloaders
from model import get_model
from train import train_model
from eval import evaluate_test

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
DATA_DIR = "data"   # Ruta donde tengas train/val/test
BATCH_SIZE = 32
EPOCHS = 5
LR = 0.001

if __name__ == "__main__":
    train_loader, val_loader, test_loader, classes = get_dataloaders(DATA_DIR, BATCH_SIZE)
    model = get_model(num_classes=len(classes)).to(DEVICE)

    model, history = train_model(model, train_loader, val_loader, DEVICE, epochs=EPOCHS, lr=LR)

    print("Evaluating on test set...")
    evaluate_test(model, test_loader, classes, DEVICE)
