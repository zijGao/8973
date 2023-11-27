import copy
import numpy as np
from tqdm import tqdm
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader,Dataset
from torch.utils.data import TensorDataset, DataLoader


def _compute_means(train_loader,val_loader,base_feature_model,know_class,total_class,feature_mean,args):
    with torch.no_grad():
        for class_idx in range(know_class, total_class):
            idx_train_loader, idx_val_loader = feature_loader(train_loader.dataset, val_loader.dataset, class_idx, args)
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
                _network(_inputs.to(args.gpu))  #extract feature
            )
        else:
            _vectors = tensor2numpy(
                _network(_inputs.to(args.gpu))
            )

        vectors.append(_vectors)
        targets.append(_targets)

    return np.concatenate(vectors), np.concatenate(targets)

#inputsize:[num_classes,512,4,4]
#[num_classes,512,4,4]->[num_classes,512*4*4]
def _compute_relations(feature_means,_known_classes):
    numpy_feature_means = np.array(feature_means)
    feature_means = numpy_feature_means.reshape(numpy_feature_means.shape[0],-1)
    old_means = np.array(feature_means[:_known_classes])
    new_means = np.array(feature_means[_known_classes:])
    _relations=np.argmax((old_means/np.linalg.norm(old_means,axis=1)[:,None])@(new_means/np.linalg.norm(new_means,axis=1)[:,None]).T,axis=1) + _known_classes
    return _relations


def _build_feature_set(_known_classes,_total_classes,feature_means,_relations,base_feature_model,train_loader,val_loader,args):
    vectors_train = []
    labels_train = []
    for class_idx in range(_known_classes, _total_classes):
        idx_train_loader, idx_val_loader = feature_loader(train_loader.dataset, val_loader.dataset, class_idx, args)
        vectors, _ = _extract_vectors(base_feature_model, idx_train_loader, args)
        vectors_train.append(vectors)
        labels_train.append([class_idx] * len(vectors))

    for class_idx in range(0,_known_classes):
        new_idx = _relations[class_idx]
        vectors_train.append(
            vectors_train[new_idx - _known_classes] - feature_means[new_idx] + feature_means[class_idx])
        labels_train.append([class_idx] * len(vectors_train[-1]))

    vectors_train = np.concatenate(vectors_train)
    labels_train = np.concatenate(labels_train)
    vectors_train_tensor =  torch.from_numpy(vectors_train)
    labels_train_tensor = torch.from_numpy(labels_train)

    # _feature_trainset = FeatureDataset(vectors_train, labels_train)
    _feature_trainset = TensorDataset(vectors_train_tensor, labels_train_tensor)
    _feature_trainset_loader = DataLoader(_feature_trainset, batch_size=args.batch_size, shuffle=True)

    vectors_test = []
    labels_test = []
    for class_idx in range(0, _total_classes):
        idx_train_loader, idx_val_loader = feature_loader(train_loader.dataset, val_loader.dataset, class_idx, args)
        vectors, _ = _extract_vectors(base_feature_model, idx_val_loader, args)
        vectors_test.append(vectors)
        labels_test.append([class_idx] * len(vectors))

    vectors_test = np.concatenate(vectors_test)
    labels_test = np.concatenate(labels_test)
    # _feature_testset = FeatureDataset(vectors_test, labels_test)
    vectors_test_tensor =  torch.from_numpy(vectors_test)
    labels_test_tensor = torch.from_numpy(labels_test)

    _feature_testset = TensorDataset(vectors_test_tensor, labels_test_tensor)
    _feature_testset_loader = DataLoader(_feature_testset, batch_size=args.batch_size, shuffle=True)


    return _feature_trainset_loader, _feature_testset_loader


def _build_current_feature_set(_known_classes,_total_classes,base_feature_model,train_loader,val_loader,args):
    base_feature_model.eval()
    vectors_train = []
    labels_train = []
    for class_idx in range(_known_classes, _total_classes):
        idx_train_loader, idx_val_loader = feature_loader(train_loader.dataset, val_loader.dataset, class_idx, args)
        vectors, _ = _extract_vectors(base_feature_model, idx_train_loader, args)
        vectors_train.append(vectors)
        labels_train.append([class_idx] * len(vectors))

    vectors_train = np.concatenate(vectors_train)
    labels_train = np.concatenate(labels_train)
    vectors_train_tensor =  torch.from_numpy(vectors_train)
    labels_train_tensor = torch.from_numpy(labels_train)

    # _feature_trainset = FeatureDataset(vectors_train, labels_train)
    _feature_trainset = TensorDataset(vectors_train_tensor, labels_train_tensor)
    _feature_trainset_loader = DataLoader(_feature_trainset, batch_size=args.batch_size, shuffle=True)

    return _feature_trainset_loader


class FeatureDataset(Dataset):
    def __init__(self, features, labels):
        assert len(features) == len(labels), "Data size error!"
        self.features = torch.from_numpy(features)
        self.labels = torch.from_numpy(labels)

    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):
        feature = self.features[idx]
        label = self.labels[idx]

        return idx, feature, label

def tensor2numpy(x):
    return x.cpu().data.numpy() if x.is_cuda else x.data.numpy()

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
    def __init__(self, projector,avgpool,flat, fc):
        super(leanrable_model, self).__init__()

        self.projector = projector
        self.avgpool = avgpool
        self.flatten = flat
        self.fc = fc
        self.fc2 = None

    def forward(self, x):
        project_feature = self.projector(x)
        project_feature = self.avgpool(project_feature)
        project_feature = self.flatten(project_feature)
        out = self.fc(project_feature)
        return project_feature, out

    def cls_forward(self, x):
        project_feature = self.projector(x)
        project_feature = self.avgpool(project_feature)
        project_feature = self.flatten(project_feature)
        out = self.fc2(project_feature)
        return out


    def freeze(self):
        for param in self.parameters():
            param.requires_grad = False
        self.eval()
        return self

    def copy(self):
        return copy.deepcopy(self)

    def extract_feature(self, x):
        project_feature = self.projector(x)
        project_feature = self.avgpool(project_feature)
        project_feature = self.flatten(project_feature)
        return project_feature


class new_model(nn.Module):
    def __init__(self, conv1,bn1,relu,layer1,layer2,layer3,layer4,projector,avgpool,flat, fc):
        super(new_model, self).__init__()
        self.conv1 = conv1
        self.bn1 = bn1
        self.relu = relu
        self.layer1 = layer1
        self.layer2 = layer2
        self.layer3 = layer3
        self.layer4 = layer4
        self.projector = projector
        self.avgpool = avgpool
        self.flatten = flat
        self.fc = fc
        self.fc2 = None

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        project_feature = self.projector(x)
        project_feature = self.avgpool(project_feature)
        project_feature = self.flatten(project_feature)
        out = self.fc(project_feature)
        return project_feature, out

    def cls_forward(self, x):
        project_feature = self.projector(x)
        project_feature = self.avgpool(project_feature)
        project_feature = self.flatten(project_feature)
        out = self.fc2(project_feature)
        return out


    def freeze(self):
        for param in self.parameters():
            param.requires_grad = False
        self.eval()
        return self

    def copy(self):
        return copy.deepcopy(self)

    def extract_feature(self, x):
        project_feature = self.projector(x)
        project_feature = self.avgpool(project_feature)
        project_feature = self.flatten(project_feature)
        return project_feature