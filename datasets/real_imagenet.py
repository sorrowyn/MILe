import glob
import json
import os
from PIL import Image
import numpy as np
import torch
import torch.nn.functional as F
import torchvision as tv

class RealImageNet(torch.utils.data.Dataset):
    def __init__(self, imagenet_path='./data/imagenet',
                 real_path='./data/real',
                 transforms=None):
        real_path = os.path.join(real_path, 'real.json')
        with open(real_path, 'r') as infile:
            self.real = json.load(infile)
        self.paths = glob.glob(os.path.join(
            imagenet_path, 'val', '*', '*.JPEG'))
        self.imagenet_path = imagenet_path
        self.paths = {path.split("/")[-1]: path for path in self.paths}
        self.real_labels = {
            f'ILSVRC2012_val_{(i+1):08d}.JPEG': labels for i, labels in enumerate(self.real)}
        self.transforms = transforms
        self.classes = os.listdir(os.path.join(imagenet_path, 'val'))
        self.classes.sort()
        self.class2idx = {clss: idx for idx, clss in enumerate(self.classes)}
        self.num_classes = 1000

    def __getitem__(self, item):
        labels = self.real_labels[f'ILSVRC2012_val_{(item + 1):08d}.JPEG']
        image_path = self.paths[f'ILSVRC2012_val_{(item + 1):08d}.JPEG']
        image_label = image_path.split("/")[-2]
        with open(image_path, 'rb') as f:
            image = Image.open(f)
            image = image.convert('RGB')
        #image = tv.io.read_image(image_path, tv.io.image.ImageReadMode.RGB)
        if self.transforms is not None:
            #image = image.float() / 255
            image = self.transforms(image)
        if len(labels) > 0:
            labels = torch.tensor(labels, dtype=torch.long)
            labels = F.one_hot(labels, self.num_classes)
            labels = labels.sum(0).float()
        else:
            labels = torch.zeros(1000, dtype=torch.float32)
        return image_path, image, self.class2idx[image_label], labels

    def __len__(self):
        return len(self.real)



class RealImageNetClasses(RealImageNet):
    def __init__(self, imagenet_path='./data/imagenet',
                 real_path='./data/real',
                 num_classes=1000,
                 seed=42,
                 transforms=None):
        super().__init__(imagenet_path, real_path, transforms)
        self.num_classes = num_classes
        if num_classes < len(self.classes):
            rng = np.random.RandomState(seed=seed)
            self.class_indices = rng.permutation(len(self.classes))[:num_classes]
            self.classes = list(sorted([self.classes[i] for i in self.class_indices]))
            self.real2idx = {real: idx for idx, real in enumerate(sorted(self.class_indices))}
            self.class2idx = {clss: idx for idx, clss in enumerate(self.classes)}
            self.indices = []
            for idx in range(super().__len__()):
                labels = self.real_labels[f'ILSVRC2012_val_{(idx + 1):08d}.JPEG']
                image_path = self.paths[f'ILSVRC2012_val_{(idx + 1):08d}.JPEG']
                image_label = image_path.split("/")[-2]
                if image_label in self.classes:
                    self.indices.append(idx)

    def __getitem__(self, item):
        item = self.indices[item]
        labels = self.real_labels[f'ILSVRC2012_val_{(item + 1):08d}.JPEG']
        image_path = self.paths[f'ILSVRC2012_val_{(item + 1):08d}.JPEG']
        image_label = image_path.split("/")[-2]
        with open(image_path, 'rb') as f:
            image = Image.open(f)
            image = image.convert('RGB')
        #image = tv.io.read_image(image_path, tv.io.image.ImageReadMode.RGB)
        if self.transforms is not None:
            #image = image.float() / 255
            image = self.transforms(image)
        labels = [self.real2idx[l] for l in labels if l in self.real2idx]
        if len(labels) > 0:
            labels = torch.tensor(labels, dtype=torch.long)
            labels = F.one_hot(labels, self.num_classes)
            labels = labels.sum(0).float()
        else:
            labels = torch.zeros(self.num_classes, dtype=torch.float32)
        return image, self.class2idx[image_label], labels

    def __len__(self):
        return len(self.indices)


class CustomRealImageNet(RealImageNet):
    def __init__(self, imagenet_path='./data/imagenet',
                 real_path='./data/real',
                 transforms=None):
        real_path = os.path.join(real_path, 'real.json')
        with open(real_path, 'r') as infile:
            self.real = json.load(infile)
        self.paths = glob.glob(os.path.join(
            imagenet_path, 'val', '*', '*.JPEG'))
        if len(self.paths) == 0:
            raise FileExistsError(os.path.join(
                imagenet_path, 'val', '*', '*.JPEG'))
        self.imagenet_path = imagenet_path
        self.paths = {path.split("/")[-1]: path for path in self.paths}
        self.indices = [int(path.split("_")[-1].split(".")[0])
                        for path in self.paths]
        self.indices.sort()
        self.real_labels = {
            f'ILSVRC2012_val_{i:08d}.JPEG': self.real[i - 1] for i in self.indices}
        self.transforms = transforms

        meta = np.load(f"{imagenet_path}/meta.npy",allow_pickle=True).item()
        self.label_map = meta['imagenet2custom']
        self.classes = os.listdir(os.path.join(imagenet_path, 'val'))
        self.classes.sort()
        self.class2idx = {clss: idx for idx, clss in enumerate(self.classes)}

    def __getitem__(self, item):
        item = self.indices[item]
        labels = self.real_labels[f'ILSVRC2012_val_{(item):08d}.JPEG']
        image_path = self.paths[f'ILSVRC2012_val_{(item):08d}.JPEG']
        image_label = image_path.split("/")[-2]
        with open(image_path, 'rb') as f:
            image = Image.open(f)
            image = image.convert('RGB')
        #image = tv.io.read_image(image_path, tv.io.image.ImageReadMode.RGB)
        if self.transforms is not None:
            #image = image.float() / 255
            image = self.transforms(image)
        labels = [self.label_map[i] for i in labels if i in self.label_map]
        if len(labels) > 0:
            labels = torch.tensor(labels, dtype=torch.long)
            labels = F.one_hot(labels, 200)
            labels = labels.sum(0).float()
        else:
            labels = torch.zeros(200, dtype=torch.float32)
        return image, self.class2idx[image_label], labels

    def __len__(self):
        return len(self.indices)


if __name__ == "__main__":
    dataset = CustomRealImageNet(imagenet_path='./data/custom_imagenet')
    pass
