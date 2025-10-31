import sys
import os
import yaml
import torch
from torch.utils.data import DataLoader, random_split
from torchvision import transforms
import dvclive
from tqdm import tqdm
import json

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from src.models import get_resnet18, get_efficientnet_b0, SmallCNNAdaptive
from src.augmentation import PlantPathologyDataset
from src.training import train_model


def train_single_model(model_name, train_loader, val_loader, dataset, train_config, device, use_augmented):
    """Обучает одну модель и возвращает результаты"""

    # Model selection
    if model_name == 'SmallCNNAdaptive':
        model = SmallCNNAdaptive(num_classes=len(dataset.labels))
    elif model_name == 'ResNet18':
        model = get_resnet18(num_classes=len(dataset.labels))
    elif model_name == 'EfficientNet-B0':
        model = get_efficientnet_b0(num_classes=len(dataset.labels))
    else:
        raise ValueError(f"Unknown model: {model_name}")

    # Training setup
    criterion = torch.nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=train_config['learning_rate'])

    # DVC Live for experiment tracking
    live_dir = "dvclive_original" if not use_augmented else "dvclive_augmented"
    with dvclive.Live(dir=live_dir) as live:
        live.log_param("model_name", model_name)
        live.log_param("use_augmented", use_augmented)

        # Train model
        train_model(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            criterion=criterion,
            optimizer=optimizer,
            num_epochs=train_config['epochs'],
            device=device,
            model_name=model_name,
            live=live
        )


def main():
    # Load parameters
    with open('params.yaml', 'r') as f:
        params = yaml.safe_load(f)

    # Configuration
    data_config = params['data']
    train_config = params['training']
    model_names = params['models']

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Data transformations
    transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    # Load dataset
    if data_config['use_augmented']:
        # Используем ОБА набора: оригиналы + аугментированные
        original_dataset = PlantPathologyDataset(
            data_config['original_csv'],
            data_config['original_img_dir'],
            transform=transform
        )
        augmented_dataset = PlantPathologyDataset(
            data_config['augmented_csv'],
            data_config['augmented_img_dir'],
            transform=transform
        )

        # Объединяем датасеты
        from torch.utils.data import ConcatDataset
        dataset = ConcatDataset([original_dataset, augmented_dataset])
        print("Using ORIGINAL + AUGMENTED data")
        print(f"Original images: {len(original_dataset)}")
        print(f"Augmented images: {len(augmented_dataset)}")
        print(f"Total images: {len(dataset)}")

    else:
        # Используем только оригиналы
        dataset = PlantPathologyDataset(
            data_config['original_csv'],
            data_config['original_img_dir'],
            transform=transform
        )
        print("Using ONLY original data")
        print(f"Total images: {len(dataset)}")

    # Split dataset
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

    train_loader = DataLoader(train_dataset, batch_size=train_config['batch_size'], shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=train_config['batch_size'], shuffle=False)

    print(f"\nDataset info:")
    print(f"Training samples: {len(train_dataset)}")
    print(f"Validation samples: {len(val_dataset)}")
    print(f"Number of classes: {len(dataset.dataset.labels) if data_config['use_augmented'] else len(dataset.labels)}")

    print(f"\nTraining {len(model_names)} models: {model_names}")

    # Прогресс-бар для обучения всех моделей
    models_pbar = tqdm(model_names, desc="Training models")

    # Train all models
    for model_name in models_pbar:
        models_pbar.set_description(f"Training {model_name}")

        print(f"\n{'=' * 60}")
        print(f"Training {model_name}...")
        print(f"{'=' * 60}")

        train_single_model(
            model_name=model_name,
            train_loader=train_loader,
            val_loader=val_loader,
            dataset=dataset.dataset if data_config['use_augmented'] else dataset,
            train_config=train_config,
            device=device,
            use_augmented=data_config['use_augmented']
        )

    print(f"\n{'=' * 60}")
    print("All models trained successfully!")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()