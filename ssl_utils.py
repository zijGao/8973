import copy
import logging
import numpy as np
from tqdm import tqdm
import torch
from torch import nn
from torch import optim
from torch.nn import functional as F
from torch.utils.data import DataLoader,Dataset

from datasets import train_loader, val_loader, train_sampler, num_classes



def _compute_means(train_loader,base_feature_model,know_class,total_class,feature_mean,args):
    with torch.no_grad():
        for class_idx in range(know_class, total_class):
            idx_train_loader, idx_val_loader = feature_loader(train_loader.dataset, val_loader.dataset, class_idx,args)
            # idx_loader = DataLoader(idx_dataset, batch_size=args.batch_size, shuffle=False, num_workers=4)
            vectors, _ = _extract_vectors(base_feature_model,idx_train_loader,args)
            class_mean = np.mean(vectors, axis=0)
            feature_mean.append(class_mean)


def feature_loader(dataset_train, dataset_val, idx, args):
    if args.dataset in ['imagenet', 'imagenet100']:
        targets_train = torch.tensor(np.array([a[1] for a in dataset_train.samples]))
        targets_val = torch.tensor(np.array([a[1] for a in dataset_val.samples]))
    else:
        targets_train = torch.tensor(dataset_train.targets)
        targets_val = torch.tensor(dataset_val.targets)

    target_idx_train = (targets_train == idx).nonzero()
    target_idx_val = (targets_val == idx).nonzero()

    dataset_train_base = torch.utils.data.Subset(dataset_train, target_idx_train)
    dataset_val_base = torch.utils.data.Subset(dataset_val, target_idx_val)

    train_loader_base = torch.utils.data.DataLoader(dataset=dataset_train_base,
                                                   batch_size=args.batch_size,
                                                   shuffle=False)
    val_loader_base = torch.utils.data.DataLoader(dataset_val_base,
                                                 batch_size=args.batch_size,
                                                 shuffle=False)
    return train_loader_base, val_loader_base

def _extract_vectors(_network, loader,args):
    _network.eval()
    vectors, targets = [], []
    for _inputs, _targets in loader:
        _targets = _targets.numpy()
        if isinstance(_network, nn.DataParallel):
            _vectors = tensor2numpy(
                _network(_inputs.to(args.gpu))[0]  #extract feature
            )
        else:
            _vectors = tensor2numpy(
                _network(_inputs.to(args.gpu))[0]
            )

        vectors.append(_vectors)
        targets.append(_targets)

    return np.concatenate(vectors), np.concatenate(targets)



def tensor2numpy(x):
    return x.cpu().data.numpy() if x.is_cuda else x.data.numpy()


def _build_feature_set(_known_classes,_total_classes,feature_means,_relations):
    vectors_train = []
    labels_train = []

    for class_idx in range(0,_known_classes):
        new_idx = _relations[class_idx]
        vectors_train.append(vectors_train[new_idx] - feature_means[new_idx] + feature_means[class_idx])
        labels_train.append([class_idx]*len(vectors_train[-1]))

    return 0

def expand_Leanrable_model_fc(model,known_class,all_class):
    new_fc = generate_fc(512, all_class * 4)

    known_output = known_class * 4
    weight = copy.deepcopy(model.fc.weight.data)
    bias = copy.deepcopy(model.fc.bias.data)
    new_fc.weight.data[:known_output] = weight
    new_fc.bias.data[:known_output] = bias
    model.fc = new_fc

def generate_fc(in_dim, out_dim):
    fc = nn.Linear(in_dim, out_dim)

    return fc


class leanrable_model(nn.Module):
    def __init__(self, feature_layer, projector, fc):
        super(leanrable_model, self).__init__()

        self.feature_extractor =  feature_layer
        self.projector = projector
        self.fc = fc
        self.fc2 = None

    def forward(self, x):
        with torch.no_grad():
            feature = self.feature_extractor(x)
        project_feature = self.projector(feature)
        out = self.fc(project_feature)
        return feature, project_feature, out

    def validate(self, x):
        with torch.no_grad():
            feature = self.feature_extractor(x)
        project_feature = self.projector(feature)
        out = self.fc2(project_feature)
        return  out

