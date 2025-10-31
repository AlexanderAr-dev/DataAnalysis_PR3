import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from torch.utils.data import Dataset, DataLoader
import torch
from PIL import Image
import albumentations as A
from albumentations.pytorch import ToTensorV2
import cv2
import yaml
import pickle  # Добавляем pickle для сохранения label_encoder


class PlantPathologyDataset(Dataset):
    def __init__(self, csv_path, images_path, transform=None, is_train=True):
        self.df = pd.read_csv(csv_path)
        self.images_path = images_path
        self.transform = transform
        self.is_train = is_train

        # Кодируем метки
        self.label_encoder = LabelEncoder()
        self.labels = self.label_encoder.fit_transform(self.df['labels'])

        # Список классов
        self.classes = self.label_encoder.classes_

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        img_name = self.df.iloc[idx]['image']
        label = self.labels[idx]

        img_path = os.path.join(self.images_path, img_name)

        # Загружаем изображение
        image = cv2.imread(img_path)
        if image is None:
            raise ValueError(f"Could not load image: {img_path}")
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        if self.transform:
            image = self.transform(image=image)['image']

        return image, label


def get_transforms(image_size=224, is_train=True):
    if is_train:
        return A.Compose([
            A.Resize(image_size, image_size),
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.5),
            A.RandomRotate90(p=0.5),
            A.ShiftScaleRotate(shift_limit=0.05, scale_limit=0.05, rotate_limit=15, p=0.5),
            A.RandomBrightnessContrast(p=0.2),
            A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ToTensorV2(),
        ])
    else:
        return A.Compose([
            A.Resize(image_size, image_size),
            A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ToTensorV2(),
        ])


def main():
    # Загружаем параметры
    with open('params.yaml', 'r') as f:
        params = yaml.safe_load(f)

    raw_path = params['data']['raw_path']
    processed_path = params['data']['processed_path']
    image_size = params['training']['image_size'][0]

    os.makedirs(processed_path, exist_ok=True)

    # Создаем datasets
    train_transform = get_transforms(image_size, is_train=True)
    val_transform = get_transforms(image_size, is_train=False)

    # Загружаем данные
    csv_path = os.path.join(raw_path, 'train.csv')
    images_path = os.path.join(raw_path, 'train_images')

    print(f"Loading data from {csv_path}")
    print(f"Images path: {images_path}")

    full_dataset = PlantPathologyDataset(csv_path, images_path, train_transform, is_train=True)

    # Разделяем на train/val
    train_idx, val_idx = train_test_split(
        range(len(full_dataset)),
        test_size=0.2,
        random_state=42,
        stratify=full_dataset.labels
    )

    # Создаем подмножества с соответствующими трансформациями
    train_dataset = torch.utils.data.Subset(full_dataset, train_idx)
    val_dataset = torch.utils.data.Subset(full_dataset, val_idx)

    # Для валидационного датасета меняем трансформации
    val_dataset.dataset.transform = val_transform

    # Сохраняем label_encoder отдельно
    with open(os.path.join(processed_path, 'label_encoder.pkl'), 'wb') as f:
        pickle.dump(full_dataset.label_encoder, f)

    # Сохраняем данные
    torch.save({
        'dataset': train_dataset,
        'classes': full_dataset.classes.tolist()  # Конвертируем в list для JSON сериализации
    }, os.path.join(processed_path, 'train_data.pt'))

    torch.save({
        'dataset': val_dataset,
        'classes': full_dataset.classes.tolist()
    }, os.path.join(processed_path, 'val_data.pt'))

    print(f"Train samples: {len(train_dataset)}")
    print(f"Val samples: {len(val_dataset)}")
    print(f"Classes: {full_dataset.classes}")
    print(f"Number of classes: {len(full_dataset.classes)}")

    # Сохраняем информацию о классах в текстовый файл для удобства
    with open(os.path.join(processed_path, 'classes.txt'), 'w') as f:
        for i, class_name in enumerate(full_dataset.classes):
            f.write(f"{i}: {class_name}\n")


if __name__ == "__main__":
    main()