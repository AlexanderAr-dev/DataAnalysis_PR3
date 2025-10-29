from torchvision import transforms
from PIL import Image
import os

augment = transforms.Compose([
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(15),
    transforms.ColorJitter(brightness=0.3, contrast=0.3)
])

src_dir = "data/train"
dst_dir = "data/train_aug"
os.makedirs(dst_dir, exist_ok=True)

for cls in os.listdir(src_dir):
    src_class = os.path.join(src_dir, cls)
    dst_class = os.path.join(dst_dir, cls)
    os.makedirs(dst_class, exist_ok=True)
    for img_name in os.listdir(src_class):
        img_path = os.path.join(src_class, img_name)
        img = Image.open(img_path)
        for i in range(2):  # две аугментации
            aug_img = augment(img)
            aug_img.save(os.path.join(dst_class, f"{img_name[:-4]}_aug{i}.jpg"))
print("✅ Augmentation complete!")
