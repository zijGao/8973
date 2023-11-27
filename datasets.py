import numpy as np
import torch
from torchvision import datasets, transforms
from main import args
import random
import copy
from uni_sampling import small_batch_dataloader
from torch.utils.data import TensorDataset
from PIL import Image
from torch.utils.data import Dataset
import os
import sys
import logging

if args.seed is not None:
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = True



def _map_new_class_index(y, order):
    return np.array(list(map(lambda x: order.index(x), y)))

def find_classes(dir):
    if sys.version_info >= (3, 5):
        # Faster and available in Python 3.5 and above
        classes = [d.name for d in os.scandir(dir) if d.is_dir()]
    else:
        classes = [d for d in os.listdir(dir) if os.path.isdir(os.path.join(dir, d))]
    classes.sort()
    class_to_idx = {classes[i]: i for i in range(len(classes))}
    return classes, class_to_idx

def pil_loader(path):
    with open(path, 'rb') as f:
        img = Image.open(f)
        return img.convert('RGB')

class DummyDataset(Dataset):
    def __init__(self, images, labels, trsf):
        assert len(images) == len(labels), 'Data size error!'
        self.images = images
        self.labels = labels
        self.trsf = trsf

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image = self.trsf(pil_loader(self.images[idx]))
        label = self.labels[idx]

        return image, label


class KuzushijiMNIST(datasets.MNIST):
    urls = [
        'http://codh.rois.ac.jp/kmnist/dataset/kmnist/train-images-idx3-ubyte.gz',
        'http://codh.rois.ac.jp/kmnist/dataset/kmnist/train-labels-idx1-ubyte.gz',
        'http://codh.rois.ac.jp/kmnist/dataset/kmnist/t10k-images-idx3-ubyte.gz',
        'http://codh.rois.ac.jp/kmnist/dataset/kmnist/t10k-labels-idx1-ubyte.gz'
    ]


kwargs = {'num_workers': args.workers, 'pin_memory': True}

if args.dataset == 'cifar100':
    input_dim = 32
    input_ch = 3
    num_classes = 100

    train_transform = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.507, 0.487, 0.441], std=[0.267, 0.256, 0.276])
    ])

    dataset_train = datasets.CIFAR100('~/data', train=True, download=True, transform=train_transform)
    dataset_val = datasets.CIFAR100('~/data', train=False,
                                        transform=transforms.Compose([
                                            transforms.ToTensor(),
                                            transforms.Normalize(mean=[0.507, 0.487, 0.441],
                                                                 std=[0.267, 0.256, 0.276])
                                        ]))
    train_loader = torch.utils.data.DataLoader(dataset=dataset_train,
                                               batch_size=args.batch_size,
                                               shuffle=True, **kwargs)
    val_loader = torch.utils.data.DataLoader(dataset_val,
                                             batch_size=args.batch_size,
                                             shuffle=False, **kwargs)

elif args.dataset == 'tiny-imagenet':
    input_dim = 64
    input_ch = 3
    num_classes = 200
    train_transform = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])
    test_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])

    #load all data
    train_dir = "/home/gao-s2r/data/tiny-imagenet-200/train"
    test_dir = "/home/gao-s2r/data/tiny-imagenet-200/val"
    train_dset = datasets.ImageFolder(train_dir)

    train_images = []
    train_labels = []
    for item in train_dset.imgs:
        train_images.append(item[0])
        train_labels.append(item[1])
    train_data, train_targets = np.array(train_images), np.array(train_labels)

    test_images = []
    test_labels = []


    _, class_to_idx = find_classes(train_dir)
    imgs_path = os.path.join(test_dir, 'images')
    imgs_annotations = os.path.join(test_dir, 'val_annotations.txt')
    with open(imgs_annotations) as r:
        data_info = map(lambda s: s.split('\t'), r.readlines())
    cls_map = {line_data[0]: line_data[1] for line_data in data_info}
    for imgname in sorted(os.listdir(imgs_path)):
        if cls_map[imgname] in sorted(class_to_idx.keys()):
            path = os.path.join(imgs_path, imgname)
            test_images.append(path)
            test_labels.append(class_to_idx[cls_map[imgname]])
    test_data, test_targets = np.array(test_images), np.array(test_labels)

    trainfolder = DummyDataset(train_data, train_targets, train_transform)
    testfolder = DummyDataset(test_data, test_targets, test_transform)

    train_loader = torch.utils.data.DataLoader(trainfolder, batch_size=args.batch_size,
                                               shuffle=True, drop_last=True, num_workers=8)

    val_loader = torch.utils.data.DataLoader(testfolder, batch_size=args.batch_size,
                                              shuffle=False, drop_last=False, num_workers=8)


elif args.dataset == 'imagenet100':
    input_dim = 224
    input_ch = 3
    num_classes = 100
    train_transform = transforms.Compose([
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
    ])
    dataset_train = datasets.ImageFolder('~/data/imagenet100/train', transform=train_transform)
    labels = np.array([a[1] for a in dataset_train.samples])
    train_loader = torch.utils.data.DataLoader(dataset=dataset_train,
                                               batch_size=args.batch_size,
                                               shuffle=True,
                                               **kwargs)

    dataset_val = datasets.ImageFolder('~/data/imagenet100/test',
                             transforms.Compose([
                                 transforms.Resize(256),
                                 transforms.CenterCrop(224),
                                 transforms.ToTensor(),
                                 transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
                             ]))
    val_loader = torch.utils.data.DataLoader(
        dataset_val,
        batch_size=args.batch_size, shuffle=False, **kwargs)

elif args.dataset == 'imagenet':
    input_dim = 224
    input_ch = 3
    num_classes = 1000
    train_transform = transforms.Compose([
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
    ])
    dataset_train = datasets.ImageFolder('~/data/imagenet/train', transform=train_transform)
    labels = np.array([a[1] for a in dataset_train.samples])
    train_loader = torch.utils.data.DataLoader(dataset=dataset_train,
                                               batch_size=args.batch_size,
                                               shuffle=True,
                                               **kwargs)

    dataset_val = datasets.ImageFolder('~/data/imagenet/val',
                             transforms.Compose([
                                 transforms.Resize(256),
                                 transforms.CenterCrop(224),
                                 transforms.ToTensor(),
                                 transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
                             ]))
    val_loader = torch.utils.data.DataLoader(
        dataset_val,
        batch_size=args.batch_size, shuffle=False, **kwargs)

else:
    print('No valid dataset is specified')



if args.distributed:
    train_sampler = torch.utils.data.distributed.DistributedSampler(dataset_train)
else:
    train_sampler = None

def get_IL_dataset(orginal_loader, IL_loader, shuffle):
    orginal_dataset = orginal_loader.dataset
    IL_dataset = IL_loader.dataset
    com_dataset = torch.utils.data.ConcatDataset([orginal_dataset, IL_dataset])
    com_loader = torch.utils.data.DataLoader(dataset=com_dataset,
                                                   batch_size=args.batch_size,
                                                   shuffle=shuffle, **kwargs)
    return com_loader


def get_IL_loader(dataset_train, dataset_val, n_classes):
    if args.dataset in ['imagenet', 'imagenet100']:
        targets_train = torch.tensor(np.array([a[1] for a in dataset_train.samples]))
        targets_val = torch.tensor(np.array([a[1] for a in dataset_val.samples]))
    elif args.dataset in ['tiny-imagenet']:
        targets_train = torch.tensor(dataset_train.labels)
        targets_val = torch.tensor(dataset_val.labels)
    else:
        targets_train = torch.tensor(dataset_train.targets)
        targets_val = torch.tensor(dataset_val.targets)
    for idx in range(args.baseclass):
        if idx == 0:
            target_idx_train = (targets_train == 0).nonzero()
            target_idx_val = (targets_val == 0).nonzero()
        else:
            target_idx_train = torch.cat((target_idx_train, (targets_train == idx).nonzero()), dim=0)
            target_idx_val = torch.cat((target_idx_val, (targets_val == idx).nonzero()), dim=0)

    dataset_train_base = torch.utils.data.Subset(dataset_train, target_idx_train)
    dataset_val_base = torch.utils.data.Subset(dataset_val, target_idx_val)

    train_loader_base = torch.utils.data.DataLoader(dataset=dataset_train_base,
                                                   batch_size=args.batch_size,
                                                   shuffle=True, **kwargs)
    val_loader_base = torch.utils.data.DataLoader(dataset_val_base,
                                                 batch_size=args.batch_size,
                                                 shuffle=True, **kwargs)

    # datasets for classes 50-99
    IL_dataset_train = []
    IL_dataset_val = []
    if args.incremental:
        if args.phase > 0:
            nc_each = (n_classes - args.baseclass) // args.phase
        else:
            nc_each = 0
        for phase in range(args.phase):
            for idx in range(args.baseclass + phase * nc_each, args.baseclass + (phase + 1) * nc_each):
                if idx == args.baseclass + phase * nc_each:
                    target_idx_train = (targets_train == args.baseclass + phase * nc_each).nonzero()
                    target_idx_val = (targets_val == args.baseclass + phase * nc_each).nonzero()
                else:
                    target_idx_train = torch.cat((target_idx_train, (targets_train == idx).nonzero()), dim=0)
                    target_idx_val = torch.cat((target_idx_val, (targets_val == idx).nonzero()), dim=0)
            dataset_train_f = torch.utils.data.Subset(dataset_train, target_idx_train)
            dataset_val_f = torch.utils.data.Subset(dataset_val, target_idx_val)

            if args.subset:
                train_loader_f = small_batch_dataloader(dataset=dataset_train_f, num_classes=nc_each,
                                                        num_samples=args.shot, batch_size=args.batch_size,
                                                        suffle=True)
            else:
                train_loader_f = torch.utils.data.DataLoader(dataset=dataset_train_f,
                                                             batch_size=args.batch_size,
                                                             shuffle=True, **kwargs)
            val_loader_f = torch.utils.data.DataLoader(dataset_val_f,
                                                       batch_size=args.batch_size,
                                                       shuffle=True, **kwargs)
            IL_dataset_train.append(train_loader_f)
            IL_dataset_val.append(val_loader_f)

    return train_loader_base, val_loader_base, IL_dataset_train, IL_dataset_val


def get_loader(dataset_train, dataset_val, n_classes):
    if args.dataset in ['imagenet', 'imagenet100']:
        targets_train = torch.tensor(np.array([a[1] for a in dataset_train.samples]))
        targets_val = torch.tensor(np.array([a[1] for a in dataset_val.samples]))
    else:
        targets_train = torch.tensor(dataset_train.targets)
        targets_val = torch.tensor(dataset_val.targets)
    for idx in range(args.baseclass):
        if idx == 0:
            target_idx_train = (targets_train == 0).nonzero()
            target_idx_val = (targets_val == 0).nonzero()
        else:
            target_idx_train = torch.cat((target_idx_train, (targets_train == idx).nonzero()), dim=0)
            target_idx_val = torch.cat((target_idx_val, (targets_val == idx).nonzero()), dim=0)

    dataset_train_base = torch.utils.data.Subset(dataset_train, target_idx_train)
    dataset_val_base = torch.utils.data.Subset(dataset_val, target_idx_val)

    train_loader_base = torch.utils.data.DataLoader(dataset=dataset_train_base,
                                                   batch_size=args.batch_size,
                                                   shuffle=True, **kwargs)
    val_loader_base = torch.utils.data.DataLoader(dataset_val_base,
                                                 batch_size=args.batch_size,
                                                 shuffle=True, **kwargs)
