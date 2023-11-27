import torch
import torchvision
import torchvision.transforms as transforms
from torchvision import datasets
import numpy as np
import matplotlib.pyplot as plt

from torch.utils.data import TensorDataset, DataLoader
from torch.utils.data.sampler import BatchSampler


class BalancedBatchSampler(BatchSampler):
    """
    BatchSampler - from a MNIST-like dataset, samples n_classes and within these classes samples n_samples.
    Returns batches of size n_classes * n_samples
    """

    def __init__(self, dataset, n_classes, n_samples):
        loader = DataLoader(dataset)
        self.labels_list = []
        for _, label in loader:
            self.labels_list.append(label)
        self.labels = torch.LongTensor(self.labels_list)
        self.labels_set = list(set(self.labels.numpy()))
        self.label_to_indices = {label: np.where(self.labels.numpy() == label)[0]
                                 for label in self.labels_set}
        for l in self.labels_set:
            np.random.shuffle(self.label_to_indices[l])
        self.used_label_indices_count = {label: 0 for label in self.labels_set}
        self.count = 0
        self.n_classes = n_classes
        self.n_samples = n_samples
        self.dataset = dataset
        self.batch_size = self.n_samples * self.n_classes

    def __iter__(self):
        self.count = 0
        while self.count + self.batch_size < len(self.dataset):
            classes = np.random.choice(self.labels_set, self.n_classes, replace=False)
            indices = []
            for class_ in classes:
                indices.extend(self.label_to_indices[class_][
                               self.used_label_indices_count[class_]:self.used_label_indices_count[
                                                                         class_] + self.n_samples])
                self.used_label_indices_count[class_] += self.n_samples
                if self.used_label_indices_count[class_] + self.n_samples > len(self.label_to_indices[class_]):
                    np.random.shuffle(self.label_to_indices[class_])
                    self.used_label_indices_count[class_] = 0
            yield indices
            self.count += self.n_classes * self.n_samples

    def __len__(self):
        return len(self.dataset) // self.batch_size

'''
n_classes = 10
n_samples = 100
batch_size = 6
mnist_train = torchvision.datasets.MNIST(root='../data/MNIST', train=True, download=True, transform=transforms.Compose([transforms.ToTensor(),]))
'''
def small_batch_dataloader(dataset=None, num_classes=10, num_samples=100, batch_size=1, suffle=True):

    balanced_batch_sampler = BalancedBatchSampler(dataset, num_classes, num_samples)
    dataloader = torch.utils.data.DataLoader(dataset, batch_sampler=balanced_batch_sampler)
    my_testiter = iter(dataloader)
    images, target = my_testiter.next()
    dataset = TensorDataset(images, target)
    del images, target
    dataloader_mini = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=suffle)

    return dataloader_mini

'''
dataloader_mini = small_batch_dataloader(dataset=mnist_train, num_classes=10, num_samples=100, batch_size=15, suffle=True)
iter_mini = iter(dataloader_mini)
images, target = iter_mini.next()
def imshow(img):
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))

imshow(torchvision.utils.make_grid(images))
'''

