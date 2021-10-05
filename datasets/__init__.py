from PIL.Image import Image
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True
from torchvision.datasets import ImageNet
import torchvision.transforms as transforms
import torchvision
from torchvision.ops import roi_align
from torchvision.transforms import functional as torchvision_F
import torchvision.datasets as datasets
import os
import torch
import numpy as np
from .real_imagenet import CustomRealImageNet, RealImageNet, RealImageNetClasses
from .imagenet_subsets import ImagenetSubset
from .synbols import SynbolsHDF5
import hashlib
import pickle
from torchvision.datasets.folder import default_loader
import pandas as pd

class ImageFolder(datasets.ImageFolder):
    def __init__(self, *args, num_classes=1000, seed=42, frac=1.,cache=True, **kwargs):
        result = hashlib.md5(args[0].encode()).hexdigest()
        filename = f"/mnt/public/tmp/{num_classes}_{seed}_{frac}_{result}.npy"
        if os.path.isfile(filename) and cache:
            with open(filename, "rb") as infile:
                self.samples = pickle.load(infile)
            self.transform = kwargs["transform"]
            self.target_transform = None
            self.loader = default_loader
        else:
            super().__init__(*args, **kwargs)
            if num_classes < len(self.classes) or frac != 1.0:
                rng = np.random.RandomState(seed=seed)
                self.class_indices = rng.permutation(len(self.classes))[:num_classes]
                # self.classes = list(sorted([self.classes[i] for i in self.class_indices]))
                # self.class_to_idx = {c: i for i, c in enumerate(self.classes)}
                self.imnet_to_cls = {idx: i for i, idx in enumerate(sorted(self.class_indices))}
                #self.samples = [(sample, self.imnet_to_cls[target]) for sample, target in self.samples if target in self.class_indices]
                rng.shuffle(self.samples)
                samples_perclass = {}
                for sample, target in self.samples:
                    if target in self.class_indices:
                        target = self.imnet_to_cls[target]
                        if target in samples_perclass:
                            samples_perclass[target].append((sample,target))
                        else:
                            samples_perclass[target] = [(sample,target)]
                samples = []
                for target in samples_perclass:
                    samples += samples_perclass[target][:int(len(samples_perclass[target])*frac)]
                self.samples = samples
                # self.new_index = rng.permutation(len(self.samples))
                # self.new_index = self.new_index[:int(len(self.new_index)*frac)]               
                self.targets = [s[1] for s in self.samples]
            if cache:
                with open(filename,"wb") as outfile:
                    pickle.dump(self.samples, outfile)

    def __getitem__(self, index):
        return tuple([index] + list(super().__getitem__(index)))

    def __len__(self):
        return len(self.samples)

class WebVision(torch.utils.data.Dataset):
    def __init__(self, data_root, train=True, transforms=None, return_weights=False):
        if train:
            meta_path_flickr = os.path.join(data_root, 'info', 'train_filelist_flickr.txt')
            meta_path_google = os.path.join(data_root, 'info', 'train_filelist_google.txt')
            # self.meta = np.loadtxt(meta_path, dtype=str, delimiter=" ")
            
            meta_flickr = pd.read_csv(meta_path_flickr, delimiter=' ', header=None).values
            meta_google = pd.read_csv(meta_path_google, delimiter=' ', header=None).values
            self.meta = np.concatenate([meta_flickr, meta_google], 0)
            self.data_root = data_root
        else:
            meta_path = os.path.join(data_root, 'info', 'val_filelist.txt')
            self.meta = pd.read_csv(meta_path, delimiter=' ', header=None).values
            # self.meta = np.loadtxt(meta_path, dtype=str, delimiter=" ")
            self.data_root = os.path.join(data_root, 'val_images')
        self.return_weights = return_weights
        if return_weights:
            self.weights = 1/np.bincount(self.meta[:, 1].astype(int))
        self.transforms = transforms

    def __getitem__(self, item):
        path, label = self.meta[item]
        path = os.path.join(self.data_root, path)
        # try:
        image = self.transforms(default_loader(path))
        # except:
        #     print(path)
        if self.return_weights:
            weight = self.weights[label]
            return item, image, label, weight
        else:
            return item, image, label
    
    def __len__(self):
        return len(self.meta)

# class ImageFolder(datasets.ImageFolder):
#     def __getitem__(self, index):
#         return tuple([index] + list(super().__getitem__(index)))

def get_labelmaps(label_maps_topk, num_batches):
    label_maps_topk_sizes = label_maps_topk[0].size()

    label_maps = torch.zeros([num_batches, 1000, label_maps_topk_sizes[2],
                              label_maps_topk_sizes[3]])
    for _label_map, _label_topk in zip(label_maps, label_maps_topk):
        _label_map = _label_map.scatter_(
            0,
            _label_topk[1][:, :, :].long(),
            _label_topk[0][:, :, :]
        )
    label_maps = label_maps.cuda()
    return label_maps


def get_relabel(label_maps, batch_coords, num_batches):
    target_relabel = roi_align(
        input=label_maps,
        boxes=torch.cat(
            [torch.arange(num_batches).view(num_batches,
                                            1).float().cuda(),
             batch_coords.float() * label_maps.size(3) - 0.5], 1),
        output_size=(1, 1))
    target_relabel = torch.nn.functional.softmax(target_relabel.squeeze(), 1)
    return target_relabel

class Multilabel_ImageFolder(datasets.ImageFolder):

    def __init__(self, root, transform, relabel_root = None, num_classes=200):
        super().__init__(root, transform)
        self.num_classes = num_classes
        if relabel_root is not None:
            self.relabel_root = relabel_root
            #self.meta = pd.read_csv('/mnt/public/datasets/WebVision/meta/train.txt', delimiter=' ', header=None).values
            meta = np.load("./data/custom_imagenet/meta.npy",allow_pickle=True).item()
            self.label_map = meta['imagenet2custom']
        
    def __getitem__(self, index):
        path, target = self.samples[index]
        sample = self.loader(path)
        if self.transform is not None:
            sample = self.transform(sample)
        if self.target_transform is not None:
            target = self.target_transform(target)
        
        split = path.split("/")
        split_pt = split[-1].split(".")

        relabel_path = os.path.join(self.relabel_root, split[-2], split_pt[0]+".pt")
        relabel_target = torch.load(relabel_path)
        relabel_target = relabel_target[1][0]
        relabel_target = torch.unique(relabel_target[2:13, 2:13])
        label = torch.zeros(self.num_classes)
        if self.num_classes < 1000:
            for l in relabel_target.view(-1):         
                if int(l) in list(self.label_map.keys()):
                    label[self.label_map[int(l)]] = 1
            if (label.sum().item() == 0):
                label[:] = 1/self.num_classes
            else:
                label = label/(label.sum())
        else:
            for l in relabel_target:
                label[int(l)] = 1
        return sample, target, label

def get_dataset(split, data_dir, exp_dict, extra="webvision"):
    if exp_dict['dataset'] == 'celeba':
        traindir = os.path.join(data_dir, exp_dict["dataset"], 'train')
        valdir = os.path.join(data_dir, exp_dict["dataset"], 'val')
        normalize = transforms.Normalize(mean=[0.50612009, 0.42543493, 0.38282761],
                                        std=[0.26589054, 0.24521921, 0.24127836])
        if split in ['train', 'train_pred']:
            transform=transforms.Compose([
                transforms.Resize(256),                  
                transforms.CenterCrop(224),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                normalize,
            ])

        elif split == 'valid':
            transform=transforms.Compose([
                transforms.Resize(256),                  
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                normalize,
            ])
        class CelebA_idx(datasets.CelebA):
            def __getitem__(self, item):
                return (item, *super().__getitem__(item)) 
        return CelebA_idx(os.path.join(data_dir, exp_dict["dataset"]), split, transform=transform) 

    if exp_dict['dataset'] == 'synbols':
        traindir = os.path.join(data_dir, "synbols", exp_dict["dataset_train"])
        testdir = os.path.join(data_dir, "synbols", exp_dict["dataset_test"])
        if split == 'train':
            train_dataset = SynbolsHDF5(traindir, ['char', 'foreground.style', 'background.color'], transforms.ToTensor(), split='train')
            return train_dataset
        elif split == 'train_pred':
            train_dataset = SynbolsHDF5(traindir, ['char', 'foreground.style', 'background.color'], transforms.ToTensor(), split='train')
            return train_dataset
        elif split == 'val':
            val_dataset = SynbolsHDF5(testdir, ['char', 'foreground.style', 'background.color'], transforms.ToTensor(), split='val')
            return val_dataset

    if exp_dict['dataset'] == 'relabel_imagenet':
        traindir = os.path.join(data_dir, "imagenet",'train')
        valdir = os.path.join(data_dir, "imagenet", 'val')
        relabeldir = os.path.join(data_dir, 'relabel/imagenet_efficientnet_l2_sz475_top5')
        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                        std=[0.229, 0.224, 0.225])
        if split == 'train':
            train_dataset = Multilabel_ImageFolder(
                traindir,
                transforms.Compose([
                    # transforms.RandomResizedCrop(224),
                    transforms.Resize(256),
                    transforms.CenterCrop(224),
                    transforms.RandomHorizontalFlip(),
                    transforms.ToTensor(),
                    normalize,
                ]), relabeldir, exp_dict["num_classes"])
            return train_dataset
        elif split == 'val':
            val_dataset = datasets.ImageFolder(
                valdir,
                transforms.Compose([
                    transforms.Resize(256),
                    transforms.CenterCrop(224),
                    transforms.ToTensor(),
                    normalize,
                ]))
            return val_dataset

    if exp_dict['dataset'] == 'custom_relabel_imagenet':
        traindir = os.path.join(data_dir, exp_dict["dataset"], 'train')
        relabeldir = os.path.join(data_dir, 'relabel/imagenet_efficientnet_l2_sz475_top5')
        valdir = os.path.join(data_dir, exp_dict["dataset"], 'val')
        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                        std=[0.229, 0.224, 0.225])
        if split == 'train':
            train_dataset = Multilabel_ImageFolder(
                traindir,
                transforms.Compose([
                    # transforms.RandomResizedCrop(224),
                    transforms.Resize(256),
                    transforms.CenterCrop(224),
                    transforms.RandomHorizontalFlip(),
                    transforms.ToTensor(),
                    normalize,
                ]), relabeldir, exp_dict["num_classes"])
            return train_dataset
        elif split == 'val':
            val_dataset = datasets.ImageFolder(
                valdir,
                transforms.Compose([
                    transforms.Resize(256),
                    transforms.CenterCrop(224),
                    transforms.ToTensor(),
                    normalize,
                ]))
            return val_dataset

    if exp_dict['dataset'] == 'custom_real_imagenet':
        valdir = os.path.join(data_dir, 'custom_imagenet')
        realdir = os.path.join(data_dir, 'real')
        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                        std=[0.229, 0.224, 0.225])
        if split == 'train': # this is a recursive function, fix it
            new_exp_dict = exp_dict.copy()
            new_exp_dict['dataset'] = 'custom_imagenet'
            return get_dataset(split, data_dir, new_exp_dict)
        elif split == 'train_pred': # this is a recursive function, fix it
            new_exp_dict_pred = exp_dict.copy()
            new_exp_dict_pred['dataset'] = 'custom_imagenet'
            return get_dataset(split, data_dir, new_exp_dict_pred)
        elif split == 'val':
            val_dataset = CustomRealImageNet(
                imagenet_path=valdir,
                real_path=realdir,
                transforms=transforms.Compose([
                    transforms.Resize(256),
                    transforms.CenterCrop(224),
                    transforms.ToTensor(),
                    normalize,
                ]))
            return val_dataset

    if exp_dict['dataset'] == 'custom_real_imagenet_fullcrop':
        valdir = os.path.join(data_dir, 'custom_imagenet')
        realdir = os.path.join(data_dir, 'real')
        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                        std=[0.229, 0.224, 0.225])
        if split == 'train': # this is a recursive function, fix it
            new_exp_dict = exp_dict.copy()
            new_exp_dict['dataset'] = 'custom_imagenet_fullcrop'
            return get_dataset(split, data_dir, new_exp_dict)
        elif split == 'train_pred': # this is a recursive function, fix it
            new_exp_dict_pred = exp_dict.copy()
            new_exp_dict_pred['dataset'] = 'custom_imagenet_fullcrop'
            return get_dataset(split, data_dir, new_exp_dict_pred)
        elif split == 'val':
            val_dataset = CustomRealImageNet(
                imagenet_path=valdir,
                real_path=realdir,
                transforms=transforms.Compose([
                    transforms.Resize(256),
                    transforms.CenterCrop(224),
                    transforms.ToTensor(),
                    normalize,
                ]))
            return val_dataset

    if exp_dict['dataset'] == 'real_imagenet':
        valdir = os.path.join(data_dir, 'imagenet')
        realdir = os.path.join(data_dir, 'real')
        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                        std=[0.229, 0.224, 0.225])
        if split == 'train': # this is a recursive function, fix it
            new_exp_dict = exp_dict.copy()
            new_exp_dict['dataset'] = 'imagenet'
            return get_dataset(split, data_dir, new_exp_dict)
        elif split == 'train_pred': # this is a recursive function, fix it
            new_exp_dict_pred = exp_dict.copy()
            new_exp_dict_pred['dataset'] = 'imagenet'
            return get_dataset(split, data_dir, new_exp_dict_pred)
        elif split == 'val':
            val_dataset = RealImageNetClasses(
                imagenet_path=valdir,
                real_path=realdir,
                transforms=transforms.Compose([
                    transforms.Resize(256),
                    transforms.CenterCrop(224),
                    transforms.ToTensor(),
                    normalize,
                ]),num_classes=exp_dict["num_classes"])
            return val_dataset


    if exp_dict['dataset'] == 'real_imagenet_full':
        valdir = os.path.join(data_dir, 'imagenet')
        realdir = os.path.join(data_dir, 'real')
        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                        std=[0.229, 0.224, 0.225])
        if split == 'train': # this is a recursive function, fix it
            new_exp_dict = exp_dict.copy()
            new_exp_dict['dataset'] = 'imagenet'
            return get_dataset(split, data_dir, new_exp_dict)
        if split == 'train_subset': # this is a recursive function, fix it
            new_exp_dict = exp_dict.copy()
            new_exp_dict['dataset'] = 'imagenet_subset'
            return get_dataset(split, data_dir, new_exp_dict)
        elif split == 'train_pred': # this is a recursive function, fix it
            new_exp_dict_pred = exp_dict.copy()
            new_exp_dict_pred['dataset'] = 'imagenet'
            return get_dataset(split, data_dir, new_exp_dict_pred)
        elif split == 'val':
            val_dataset = RealImageNet(
                imagenet_path=valdir,
                real_path=realdir,
                transforms=transforms.Compose([
                    transforms.Resize(256),
                    transforms.CenterCrop(224),
                    transforms.ToTensor(),
                    normalize,
                ]))
            return val_dataset

    if exp_dict['dataset'] == 'webvision' and extra == "imagenet":
        valdir = os.path.join(data_dir, 'imagenet')
        realdir = os.path.join(data_dir, 'real')
        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                        std=[0.229, 0.224, 0.225])
        if split == 'train': # this is a recursive function, fix it
            new_exp_dict = exp_dict.copy()
            new_exp_dict['dataset'] = 'imagenet'
            return get_dataset(split, data_dir, new_exp_dict)
        if split == 'train_subset': # this is a recursive function, fix it
            new_exp_dict = exp_dict.copy()
            new_exp_dict['dataset'] = 'imagenet_subset'
            return get_dataset(split, data_dir, new_exp_dict)
        elif split == 'train_pred': # this is a recursive function, fix it
            new_exp_dict_pred = exp_dict.copy()
            new_exp_dict_pred['dataset'] = 'imagenet'
            return get_dataset(split, data_dir, new_exp_dict_pred)
        elif split == 'val':
            val_dataset = RealImageNet(
                imagenet_path=valdir,
                real_path=realdir,
                transforms=transforms.Compose([
                    transforms.Resize(256),
                    transforms.CenterCrop(224),
                    transforms.ToTensor(),
                    normalize,
                ]))
            return val_dataset


    if exp_dict['dataset'] == 'real_imagenet_fullcrop':
        valdir = os.path.join(data_dir, 'imagenet')
        realdir = os.path.join(data_dir, 'real')
        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                        std=[0.229, 0.224, 0.225])
        if split == 'train': # this is a recursive function, fix it
            new_exp_dict = exp_dict.copy()
            new_exp_dict['dataset'] = 'imagenet_fullcrop'
            return get_dataset(split, data_dir, new_exp_dict)
        elif split == 'train_pred': # this is a recursive function, fix it
            new_exp_dict_pred = exp_dict.copy()
            new_exp_dict_pred['dataset'] = 'imagenet_fullcrop'
            return get_dataset(split, data_dir, new_exp_dict_pred)
        elif split == 'val':
            val_dataset = RealImageNetClasses(
                imagenet_path=valdir,
                real_path=realdir,
                transforms=transforms.Compose([
                    transforms.Resize(256),
                    transforms.CenterCrop(224),
                    transforms.ToTensor(),
                    normalize,
                ]),num_classes=exp_dict["num_classes"])
            return val_dataset

    if exp_dict['dataset'] == 'custom_imagenet':
        traindir = os.path.join(data_dir, exp_dict["dataset"], 'train')
        valdir = os.path.join(data_dir, exp_dict["dataset"], 'val')
        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                        std=[0.229, 0.224, 0.225])
        if split == 'train':
            train_dataset = ImageFolder(
                traindir,
                transforms.Compose([
                    transforms.RandomResizedCrop(224),
                    #transforms.CenterCrop(224),
                    transforms.RandomHorizontalFlip(),
                    transforms.ToTensor(),
                    normalize,
                ]),
                num_classes=exp_dict["num_classes"])
            return train_dataset
        elif split == 'train_pred':
            train_dataset = ImageFolder(
                traindir,
                transforms.Compose([
                    # transforms.RandomResizedCrop(224),
                    transforms.Resize(224),
                    transforms.CenterCrop(224),
                    # transforms.RandomHorizontalFlip(),
                    transforms.ToTensor(),
                    normalize,
                ]),
                num_classes=exp_dict["num_classes"])
            return train_dataset
        elif split == 'val':
            val_dataset = ImageFolder(
                valdir,
                transforms.Compose([
                    transforms.Resize(256),
                    transforms.CenterCrop(224),
                    transforms.ToTensor(),
                    normalize,
                ]),
                num_classes=exp_dict["num_classes"])
            return val_dataset

    if exp_dict['dataset'] == 'custom_imagenet_fullcrop':
        traindir = os.path.join(data_dir, exp_dict["dataset"], 'train')
        valdir = os.path.join(data_dir, exp_dict["dataset"], 'val')
        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                        std=[0.229, 0.224, 0.225])
        if split == 'train':
            train_dataset = ImageFolder(
                traindir,
                transforms.Compose([
                    # transforms.RandomResizedCrop(224),                  
                    transforms.Resize(256),
                    transforms.RandomCrop(224),
                    transforms.RandomHorizontalFlip(),
                    transforms.ToTensor(),
                    normalize,
                ]),
                num_classes=exp_dict["num_classes"])
            return train_dataset
        elif split == 'train_pred':
            train_pred_dataset = ImageFolder(
                traindir,
                transforms.Compose([
                    # transforms.RandomResizedCrop(224),                  
                    transforms.Resize(256),
                    transforms.RandomCrop(224),
                    transforms.RandomHorizontalFlip(),
                    transforms.ToTensor(),
                    normalize,
                ]),
                num_classes=exp_dict["num_classes"])
            return train_pred_dataset
        elif split == 'val':
            val_dataset = datasets.ImageFolder(
                valdir,
                transforms.Compose([
                    transforms.Resize(256),
                    transforms.CenterCrop(224),
                    transforms.ToTensor(),
                    normalize,
                ]),
                num_classes=exp_dict["num_classes"])
            return val_dataset

    if exp_dict['dataset'] == 'webvision':
        data_root = os.path.join(data_dir, exp_dict["dataset"])
        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                        std=[0.229, 0.224, 0.225])
        if split == 'train':
            train_dataset = WebVision(
                data_root,
                train=True,
                transforms=transforms.Compose([
                    transforms.RandomResizedCrop(224),                  
                    #transforms.CenterCrop(224),
                    transforms.RandomHorizontalFlip(),
                    transforms.ToTensor(),
                    normalize,
                ]),
                return_weights=exp_dict.get('reweight', '') != '')
            return train_dataset

        elif split == 'train_pred':
            train_dataset = WebVision(
                data_root,
                train=True,
                transforms=transforms.Compose([
                    # transforms.RandomResizedCrop(224),
                    transforms.Resize(224),
                    transforms.CenterCrop(224),
                    # transforms.RandomHorizontalFlip(),
                    transforms.ToTensor(),
                    normalize,
                ]),
                return_weights=exp_dict.get('reweight', '') != '')
            return train_dataset
        elif split == 'val':
            val_dataset = WebVision(
                data_root,
                train=False,
                transforms=transforms.Compose([
                    transforms.Resize(256),
                    transforms.CenterCrop(224),
                    transforms.ToTensor(),
                    normalize,
                ]))
            return val_dataset

    if exp_dict['dataset'] == 'imagenet':
        traindir = os.path.join(data_dir, exp_dict["dataset"], 'train')
        valdir = os.path.join(data_dir, exp_dict["dataset"], 'val')
        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                        std=[0.229, 0.224, 0.225])
        if split == 'train':
            train_dataset = ImageFolder(
                traindir,
                transform=transforms.Compose([
                    transforms.RandomResizedCrop(224),                  
                    #transforms.CenterCrop(224),
                    transforms.RandomHorizontalFlip(),
                    transforms.ToTensor(),
                    normalize,
                ]),
                num_classes=exp_dict["num_classes"], frac=exp_dict["train_frac"])
            return train_dataset

        elif split == 'train_pred':
            train_dataset = ImageFolder(
                traindir,
                transform=transforms.Compose([
                    # transforms.RandomResizedCrop(224),
                    transforms.Resize(224),
                    transforms.CenterCrop(224),
                    # transforms.RandomHorizontalFlip(),
                    transforms.ToTensor(),
                    normalize,
                ]),
                num_classes=exp_dict["num_classes"])
            return train_dataset
        elif split == 'val':
            val_dataset = datasets.ImageFolder(
                valdir,
                transform=transforms.Compose([
                    transforms.Resize(256),
                    transforms.CenterCrop(224),
                    transforms.ToTensor(),
                    normalize,
                ]),
                num_classes=exp_dict["num_classes"])
            return val_dataset

    if exp_dict['dataset'] == 'imagenet_subset':
        traindir = os.path.join(data_dir, 'imagenet', 'train')
        valdir = os.path.join(data_dir, 'imagenet', 'val')
        subset_path = os.path.join(data_dir, exp_dict['subset_path']) # path to the text file (relative to data_dir)
        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                        std=[0.229, 0.224, 0.225])
        if split == 'train_subset':
            train_dataset = ImagenetSubset(
                traindir,
                subset_path=subset_path,
                transform=transforms.Compose([
                    transforms.RandomResizedCrop(224),                  
                    #transforms.CenterCrop(224),
                    transforms.RandomHorizontalFlip(),
                    transforms.ToTensor(),
                    normalize,
                ]))
            return train_dataset

        elif split == 'train_subset_pred':
            train_dataset = ImagenetSubset(
                traindir,
                transform=transforms.Compose([
                    # transforms.RandomResizedCrop(224),
                    transforms.Resize(224),
                    transforms.CenterCrop(224),
                    # transforms.RandomHorizontalFlip(),
                    transforms.ToTensor(),
                    normalize,
                ]),
                num_classes=exp_dict["num_classes"])
            return train_dataset
        elif split == 'val':
            val_dataset = datasets.ImageFolder(
                valdir,
                transform=transforms.Compose([
                    transforms.Resize(256),
                    transforms.CenterCrop(224),
                    transforms.ToTensor(),
                    normalize,
                ]),
                num_classes=exp_dict["num_classes"])
            return val_dataset

    if exp_dict['dataset'] == 'imagenet_fullcrop':
        traindir = os.path.join(data_dir, exp_dict["dataset"], 'train')
        valdir = os.path.join(data_dir, exp_dict["dataset"], 'val')
        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                        std=[0.229, 0.224, 0.225])
        if split == 'train':
            train_dataset = ImageFolder(
                traindir,
                transforms.Compose([
                    # transforms.RandomResizedCrop(224),                  
                    transforms.Resize(256),
                    transforms.RandomCrop(224),
                    transforms.RandomHorizontalFlip(),
                    transforms.ToTensor(),
                    normalize,
                ]),
                num_classes=exp_dict["num_classes"])
            return train_dataset
        elif split == 'train_pred':
            train_pred_dataset = ImageFolder(
                traindir,
                transforms.Compose([
                    # transforms.RandomResizedCrop(224),                  
                    transforms.Resize(256),
                    transforms.RandomCrop(224),
                    transforms.RandomHorizontalFlip(),
                    transforms.ToTensor(),
                    normalize,
                ]),
                num_classes=exp_dict["num_classes"])
            return train_pred_dataset
        elif split == 'val':
            val_dataset = datasets.ImageFolder(
                valdir,
                transforms.Compose([
                    transforms.Resize(256),
                    transforms.CenterCrop(224),
                    transforms.ToTensor(),
                    normalize,
                ]),
                num_classes=exp_dict["num_classes"])
            return val_dataset

    elif exp_dict['dataset'] == 'cifar10':
        transform_train = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ])

        transform_val = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ])
        if split == 'train':
            train_dataset = torchvision.datasets.CIFAR10(
                root=data_dir, train=True, download=True, transform=transform_train)
            return train_dataset
        elif split == 'val':
            val_dataset = torchvision.datasets.CIFAR10(
                root=data_dir, train=False, download=True, transform=transform_val)
            return val_dataset

    elif exp_dict['dataset'] == 'imagenet-200':
        traindir = os.path.join(data_dir, 'train')
        valdir = os.path.join(data_dir, 'val')
        transform_train = transforms.Compose([
            transforms.RandomResizedCrop(64, scale=(0.8, 1.0)),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
        ])

        transform_val = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], 
                                 [0.229, 0.224, 0.225]),
        ])
        if split == 'train':
            train_dataset = torchvision.datasets.ImageFolder(
                root=traindir, transform=transform_train)
            return train_dataset
        elif split == 'val':
            val_dataset = torchvision.datasets.ImageFolder(
                root=valdir, transform=transform_val)
            return val_dataset

    raise NotImplementedError