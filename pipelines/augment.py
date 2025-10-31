import sys
import os

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from src.augmentation import augment_data


def main():
    print("Starting data augmentation...")

    augment_data(
        input_csv="data/raw/train.csv",
        input_img_dir="data/raw/train_images",
        output_csv="data/augmented/train_augmented.csv",
        output_img_dir="data/augmented/train",
        num_augmentations=1
    )

    print("Data augmentation completed!")


if __name__ == "__main__":
    main()