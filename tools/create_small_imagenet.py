import numpy as np
import os
import shutil as sh
from tqdm import tqdm
import json
import glob

np.random.seed(42)
if __name__ == "__main__":
    original_imagenet_path = '/mnt/public/datasets/imagenet/raw/'
    classes = glob.glob('/mnt/public/datasets/imagenet/raw/train/*/')
    classes = [c.split('/')[-2] for c in classes]
    root = f"/mnt/public/datasets/imagenet/"
    n_classes = 200
    images_per_class = 100
    dataset_name = f"custom_imagenet_{n_classes}"
    train_path = os.path.join(root, dataset_name, 'train')
    val_path = os.path.join(root, dataset_name, 'val')
    os.makedirs(train_path, exist_ok=True)
    os.makedirs(val_path, exist_ok=True)
    classes = list(sorted(classes))
    subset = list(sorted(np.random.choice(classes, n_classes, replace=False)))
    imagenet2custom = {}
    for i, c in enumerate(subset):
        imagenet2custom[classes.index(c)] = i
    np.save(f"/mnt/public/datasets/imagenet/{dataset_name}/meta.npy", {'imagenet2custom': imagenet2custom, 'subset': subset})

    for subset_class in tqdm(subset):
        os.makedirs(os.path.join(train_path, subset_class), exist_ok=True)
        os.makedirs(os.path.join(val_path, subset_class), exist_ok=True)
        train_images = list(sorted(os.listdir(os.path.join(original_imagenet_path, 'train', subset_class))))
        train_subset = np.random.choice(train_images, images_per_class)
        for im in train_subset:
            src_path = os.path.join(original_imagenet_path, 'train', subset_class, im)
            dst_path = os.path.join(train_path, subset_class, im)
            sh.copy2(src_path, dst_path)
        
        sh.copytree(os.path.join(original_imagenet_path, 'val', subset_class), os.path.join(val_path, subset_class), dirs_exist_ok=True)