from torch.utils.data import Dataset
import os
import torchvision

class ImagenetSubset(Dataset):
    def __init__(self, data_root, subset_path, transform=None):
        self.data_root = data_root
        with open(subset_path, 'r') as infile:
            inlines = infile.read()
            self.subset_images = inlines.split('\n')
        self.transform = transform
        self.targets_set = os.listdir(data_root)
        self.target2idx = {target: i for i, target in enumerate(self.targets_set)}
        self.labels = [self.target2idx[p.split("_")[0]] for p in self.subset_images]
        self.loader = torchvision.datasets.folder.default_loader

    def __getitem__(self, item):
        path = os.path.join(self.data_root, self.subset_images[item].split("_")[0], self.subset_images[item])
        label = self.labels[item]
        sample = self.loader(path)
        if self.transform is not None:
            sample = self.transform(sample)

        return item, sample, label

    def __len__(self):
        return len(self.subset_images)