"""
MAI/IDL SS26 - Final assignment. 

MG 6/6/2026
"""
import torch
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score
)

class Trainer:
    def __init__(self, model, criterion, optimizer, device):
        self.model = model
        self.criterion = criterion
        self.optimizer = optimizer
        self.device = device

    def train_one_epoch(self, dataloader):
        self.model.train()
        running_loss = 0.0
        correct, total = 0, 0
        
        for images, labels in dataloader:
            images, labels = images.to(self.device), labels.to(self.device)
            
            outputs = self.model(images)
            loss = self.criterion(outputs, labels)
            self.optimizer.zero_grad() #Fix: Clear gradients before backward pass
            loss.backward()
            self.optimizer.step()
            
            running_loss += loss.item() * images.size(0)
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
            
        return running_loss / total, (correct / total) * 100

    def evaluate(self, dataloader):
        self.model.eval()

        running_loss = 0.0
        total = 0

        all_predictions = []
        all_labels = []

        with torch.no_grad():
            for images, labels in dataloader:

                images = images.to(self.device)
                labels = labels.to(self.device)

                outputs = self.model(images)

                loss = self.criterion(outputs, labels)

                running_loss += loss.item() * images.size(0)

                _, predicted = outputs.max(1)

                total += labels.size(0)

                all_predictions.extend(predicted.cpu().tolist())
                all_labels.extend(labels.cpu().tolist())

        metrics = {
            "loss": running_loss / total,
            "accuracy": accuracy_score(
                all_labels,
                all_predictions
            ) * 100,
            "precision": precision_score(
                all_labels,
                all_predictions,
                average="macro",
                zero_division=0
            ) * 100,
            "recall": recall_score(
                all_labels,
                all_predictions,
                average="macro",
                zero_division=0
            ) * 100,
            "macro_f1": f1_score(
                all_labels,
                all_predictions,
                average="macro",
                zero_division=0
            ) * 100,
        }

        return metrics

    def fit(self, train_loader, val_loader, epochs):
        print("\n Starting Training Routine...")
        print("-" * 50)
        
        for epoch in range(epochs):
            train_loss, train_acc = self.train_one_epoch(train_loader)
            val_metrics = self.evaluate(val_loader)
            
            print(f"Epoch [{epoch+1:02d}/{epochs:02d}] | "
                  f"Train Loss: {train_loss:.4f} - Train Acc: {train_acc:.2f}% | "
                  f"Val Loss: {val_metrics['loss']:.4f} - Val Acc: {val_metrics['accuracy']:.2f}%")
        
        print("-" * 50)
        print("Training Complete!")
