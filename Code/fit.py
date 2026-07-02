"""
MAI/IDL SS26 - Final assignment. 

MG 6/6/2026
"""
import torch
from pathlib import Path
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score
)
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

class Trainer:
    def __init__(self, model, criterion, optimizer, device, save_model=False, model_path="best_model.pt"):
        self.model = model
        self.criterion = criterion
        self.optimizer = optimizer
        self.device = device
        self.save_model = save_model
        self.model_path = model_path
        if self.save_model:
            Path(self.model_path).parent.mkdir(
                parents=True,
                exist_ok=True
            )

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
        print("\nStarting Training Routine...")
        print("-" * 50)
        best_val_acc = 0.0
        history = {"train_loss": [], "val_loss": []}
        for epoch in range(epochs):
            train_loss, train_acc = self.train_one_epoch(train_loader)
            val_metrics = self.evaluate(val_loader)

            history["train_loss"].append(train_loss)
            history["val_loss"].append(val_metrics["loss"])

            print(f"Epoch [{epoch+1:02d}/{epochs:02d}] | "
                  f"Train Loss: {train_loss:.4f} - Train Acc: {train_acc:.2f}% | "
                  f"Val Loss: {val_metrics['loss']:.4f} - Val Acc: {val_metrics['accuracy']:.2f}%")

            if self.save_model and val_metrics['accuracy'] > best_val_acc:
                best_val_acc = val_metrics['accuracy']

                torch.save(
                    self.model.state_dict(),
                    self.model_path
                )

                print(
                    f"New best model saved "
                    f"(Val Acc: {val_metrics['accuracy']:.2f}%)"
                )
        print("-" * 50)
        print("Training Complete!")

        return history
