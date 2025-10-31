import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
import numpy as np
import dvclive
import os
from tqdm import tqdm


def train_model(model, train_loader, val_loader, criterion, optimizer, num_epochs, device, model_name, live=None):
    model = model.to(device)
    best_val_loss = float('inf')
    best_metrics = {}

    for epoch in range(num_epochs):
        # Training phase
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0

        # Прогресс-бар для обучения
        train_pbar = tqdm(train_loader, desc=f'{model_name} Epoch {epoch + 1}/{num_epochs} [Train]')

        for images, labels in train_pbar:
            images, labels = images.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            train_total += labels.size(0)
            train_correct += (torch.sigmoid(outputs).round() == labels).float().mean().item()

            # Обновляем прогресс-бар
            train_pbar.set_postfix({
                'Loss': f'{loss.item():.4f}',
                'Acc': f'{(torch.sigmoid(outputs).round() == labels).float().mean().item():.4f}'
            })

        # Validation phase
        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0
        all_preds = []
        all_labels = []

        # Прогресс-бар для валидации
        val_pbar = tqdm(val_loader, desc=f'{model_name} Epoch {epoch + 1}/{num_epochs} [Val]')

        with torch.no_grad():
            for images, labels in val_pbar:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                loss = criterion(outputs, labels)

                val_loss += loss.item()
                val_total += labels.size(0)
                val_correct += (torch.sigmoid(outputs).round() == labels).float().mean().item()

                all_preds.extend(torch.sigmoid(outputs).round().cpu().numpy())
                all_labels.extend(labels.cpu().numpy())

                # Обновляем прогресс-бар
                val_pbar.set_postfix({
                    'Loss': f'{loss.item():.4f}',
                    'Acc': f'{(torch.sigmoid(outputs).round() == labels).float().mean().item():.4f}'
                })

        # Calculate metrics
        train_acc = train_correct / len(train_loader)
        val_acc = val_correct / len(val_loader)

        all_preds = np.array(all_preds)
        all_labels = np.array(all_labels)

        f1 = f1_score(all_labels, all_preds, average='macro', zero_division=0)
        precision = precision_score(all_labels, all_preds, average='macro', zero_division=0)
        recall = recall_score(all_labels, all_preds, average='macro', zero_division=0)

        # Log metrics
        if live:
            live.log_metric(f"{model_name}/train/loss", train_loss / len(train_loader))
            live.log_metric(f"{model_name}/train/accuracy", train_acc)
            live.log_metric(f"{model_name}/val/loss", val_loss / len(val_loader))
            live.log_metric(f"{model_name}/val/accuracy", val_acc)
            live.log_metric(f"{model_name}/val/f1_score", f1)
            live.log_metric(f"{model_name}/val/precision", precision)
            live.log_metric(f"{model_name}/val/recall", recall)
            live.next_step()

        print(f'\n{model_name} - Epoch [{epoch + 1}/{num_epochs}] Summary:')
        print(f'Train Loss: {train_loss / len(train_loader):.4f}, Train Acc: {train_acc:.4f}')
        print(f'Val Loss: {val_loss / len(val_loader):.4f}, Val Acc: {val_acc:.4f}')
        print(f'Val F1: {f1:.4f}, Precision: {precision:.4f}, Recall: {recall:.4f}')
        print('-' * 60)

        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_metrics = {
                'val_loss': val_loss / len(val_loader),
                'val_accuracy': val_acc,
                'val_f1': f1,
                'val_precision': precision,
                'val_recall': recall
            }
            os.makedirs('models', exist_ok=True)
            torch.save(model.state_dict(), f'models/{model_name}_best.pth')

    print(f"\nBest metrics for {model_name}:")
    for metric, value in best_metrics.items():
        print(f"  {metric}: {value:.4f}")
    return best_metrics