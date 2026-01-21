from pathlib import Path

DATASET_PATH = Path("/Users/prathamkumarpatel/Documents/Projects/plant_diseases_project/data_raw/PlantVillage")

classes = [p for p in DATASET_PATH.iterdir() if p.is_dir()]

print("Dataset path:", DATASET_PATH)
print("Total classes:", len(classes))
