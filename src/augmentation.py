import os
import pandas as pd
from PIL import Image
import torch
from torchvision import transforms
from torch.utils.data import Dataset
from torchvision.utils import save_image
from tqdm import tqdm


class PlantPathologyDataset(Dataset):
    def __init__(self, csv_file, img_dir, transform=None):
        self.data = pd.read_csv(csv_file)
        self.img_dir = img_dir
        self.transform = transform
        self.labels = self.get_unique_labels()

    def get_unique_labels(self):
        return sorted(set(label for labels in self.data['labels'] for label in labels.split()))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img_name = self.data.iloc[idx]['image']
        img_path = os.path.join(self.img_dir, img_name)
        image = Image.open(img_path).convert('RGB')
        label = self.data.iloc[idx]['labels']

        if self.transform:
            image = self.transform(image)

        # Multi-label encoding
        label_encoded = torch.zeros(len(self.labels))
        for lbl in label.split():
            if lbl in self.labels:
                label_encoded[self.labels.index(lbl)] = 1

        return image, label_encoded


def augment_data(input_csv, input_img_dir, output_csv, output_img_dir, num_augmentations=1):
    """
    Аугментация данных - создает ТОЛЬКО аугментированные версии
    """
    os.makedirs(output_img_dir, exist_ok=True)

    # Трансформации для аугментации
    augment_transform = transforms.Compose([
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomVerticalFlip(p=0.3),
        transforms.RandomRotation(degrees=15),
        transforms.Resize((256, 256)),  # Убрали сложные трансформации для скорости
        transforms.ToTensor(),
    ])

    data = pd.read_csv(input_csv)
    augmented_data = []

    print(f"Starting augmentation...")
    print(f"Original images: {len(data)}")
    print(f"Target augmented images: {len(data) * num_augmentations}")

    # Прогресс-бар для аугментации
    with tqdm(total=len(data), desc="Augmenting images") as pbar:
        for idx, row in data.iterrows():
            img_name = row['image']
            label = row['labels']
            img_path = os.path.join(input_img_dir, img_name)

            # Оригинальное изображение
            original_img = Image.open(img_path).convert('RGB')

            # Создаем ТОЛЬКО аугментированные версии (оригиналы не копируем!)
            for aug_idx in range(num_augmentations):
                augmented_img = augment_transform(original_img)
                aug_img_name = f"aug_{aug_idx}_{img_name}"
                save_image(augmented_img, os.path.join(output_img_dir, aug_img_name))
                augmented_data.append({'image': aug_img_name, 'labels': label})

            pbar.update(1)
            pbar.set_postfix({
                'processed': f"{idx + 1}/{len(data)}",
                'total_created': len(augmented_data)
            })

    # Сохраняем CSV только с аугментированными изображениями
    augmented_df = pd.DataFrame(augmented_data)
    augmented_df.to_csv(output_csv, index=False)

    print(f"\nAugmentation completed!")
    print(f"Created {len(augmented_df)} augmented images")
    print(f"Original images: {len(data)} (используются напрямую)")
    print(f"Total for training: {len(data) + len(augmented_df)} images")
    return augmented_df