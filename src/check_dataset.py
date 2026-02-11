#version 0.1
from pathlib import Path

DATASET_PATH = Path("/Users/prathamkumarpatel/Documents/Projects/plant_diseases_project/data_raw/PlantVillage")

classes = [p for p in DATASET_PATH.iterdir() if p.is_dir()]

print("Dataset path:", DATASET_PATH)
print("Total classes:", len(classes))


image_counts = {}

for class_dir in classes:
    image_files = [
        f for f in class_dir.iterdir()
        if f.suffix.lower() in [".jpg", ".jpeg", ".png"]
    ]
    image_counts[class_dir.name] = len(image_files)

total_images = sum(image_counts.values())
print("Total images:", total_images)


sorted_classes = sorted(
    image_counts.items(),
    key=lambda x: x[1],
    reverse=True
)

print("\nTop 5 classes:")
for cls, count in sorted_classes[:5]:
    print(cls, ":", count)

print("\nBottom 5 classes:")
for cls, count in sorted_classes[-5:]:
    print(cls, ":", count)


max_count = max(image_counts.values())
min_count = min(image_counts.values())

imbalance_ratio = max_count / min_count
print("\nImbalance ratio (max/min):", round(imbalance_ratio, 2))




from PIL import Image
import random

sample_classes = random.sample(classes, 3)

for class_dir in sample_classes:
    images = list(class_dir.iterdir())
    img_path = random.choice(images)
    
    with Image.open(img_path) as img:
        print(f"{class_dir.name} → size: {img.size}")



import matplotlib.pyplot as plt

class_names = [cls for cls, _ in sorted_classes]
counts = [count for _, count in sorted_classes]

plt.figure(figsize=(12, 5))
plt.bar(class_names, counts)
plt.xticks(rotation=90)
plt.title("Class Distribution (PlantVillage)")
plt.xlabel("Class")
plt.ylabel("Number of Images")
plt.tight_layout()
plt.show()


