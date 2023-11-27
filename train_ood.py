import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Dataset
from torch.utils.data import ConcatDataset
from torch.utils.data import DataLoader, TensorDataset
from parsers import args
import shutil
from enum import Enum
import time
import os
import torch.nn.functional as F


print('creating model saving dir!')
dirname = ''
dirname = os.path.join(dirname, 'save_ood_model')
dirname = os.path.join(dirname, args.dataset)
dirname = os.path.join(dirname, args.arch)
dirname = dirname.replace("\\", "/")
import datetime
import numpy as np
now = datetime.datetime.now()
string = str(now.month) + '-' + str(now.day) + '_' + str(now.hour) + '-' + str(
    now.minute) + '-' + str(now.second) + '-' + str(np.random.randint(1000))
dirname = os.path.join(dirname, string)
args.dirname = dirname
print('saving to: {}'.format(args.dirname))
if not os.path.exists(args.dirname):
    os.makedirs(args.dirname)
with open(os.path.join(args.dirname, 'args.txt'), 'w') as file:
    file.write(str(args))

memory_size = 2048  # 定义memory的长度

from datasets import train_loader, val_loader, train_sampler, num_classes

# num_classes = 100
args.full_classes = num_classes

from datasets import get_IL_loader

train_loader_base, val_loader_base, \
IL_dataset_train, IL_dataset_val, = get_IL_loader(train_loader.dataset, val_loader.dataset, num_classes)
num_classes = args.baseclass
train_loader, val_loader = train_loader_base, val_loader_base

import torch
from torch import nn, optim

ngpus_per_node = torch.cuda.device_count()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.cuda.set_device(args.gpu)

#与ACIL同步进行： epoch 160 lr 0.001
#1、backbone + n 个head（n为任务数，head输出层为num—classes 100）
#2、在任务n时，训练任务n的backbone+head，数据来源：当前data+memory（第一个任务没有），然后更新memory
#3、每个任务训练完之后back-update之前n-1个head，backbone不变

#或者
#不建议3、在任务n时，训练完之后接着用memory(任务1-任务n的样本)来分别back-update所有分类器，第0个任务可用memory0-9，第1个任务可用memory1-9，以此类推
#4、推理阶段，所有head的对应类的神经元分类值拼接在一起，100个类里面直接argmax挑前10个类别，ACIL对应的10个神经元中找权重最大的当分类结果
#或者
#4、先OOD检测属于第n个任务，ACIL对应的第n个任务中，找权重最大的当结果

#以下类及函数部分为老师代码中的，不用修改

def accuracy(output, target, topk=(1,)):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res

def validate(val_loader, model, criterion, args, print=True):
    batch_time = AverageMeter('Time', ':6.3f', Summary.NONE)
    losses = AverageMeter('Loss', ':.4e', Summary.NONE)
    top1 = AverageMeter('Acc@1', ':6.2f', Summary.AVERAGE)
    top5 = AverageMeter('Acc@5', ':6.2f', Summary.AVERAGE)
    progress = ProgressMeter(
        len(val_loader),
        [batch_time, losses, top1, top5],
        prefix='Test: ')

    # switch to evaluate mode
    model.eval()

    with torch.no_grad():
        end = time.time()
        for i, (images, target) in enumerate(val_loader):
            if args.gpu is not None:
                images = images.cuda(args.gpu, non_blocking=True)
            if torch.cuda.is_available():
                target = target.cuda(args.gpu, non_blocking=True)

            # compute output
            output = model(images)
            if isinstance(criterion, nn.MSELoss):
                target_onehot = F.one_hot(target, args.num_classes).float()
            else:
                target_onehot = target

            loss = criterion(output, target_onehot)

            # measure accuracy and record loss
            acc1, acc5 = accuracy(output, target, topk=(1, 5))
            losses.update(loss.item(), images.size(0))
            top1.update(acc1[0], images.size(0))
            top5.update(acc5[0], images.size(0))

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            if i % args.print_freq == 0:
                progress.display(i)

        progress.display_summary()
    if print:
        with open(os.path.join(args.dirname, 'args.txt'), 'a+') as file:
            file.write(str(round(top1.avg.item(), 4)) + '\n')

    return top1.avg


def train(train_loader, model, criterion, optimizer, epoch, args):
    batch_time = AverageMeter('Time', ':6.3f')
    data_time = AverageMeter('Data', ':6.3f')
    losses = AverageMeter('Loss', ':.4e')
    top1 = AverageMeter('Acc@1', ':6.2f')
    top5 = AverageMeter('Acc@5', ':6.2f')
    LR = AverageMeter('lr', ':6.3f')  # todo: val not avg
    progress = ProgressMeter(
        len(train_loader),
        [batch_time, data_time, losses, top1, top5, LR],
        prefix="Epoch: [{}]".format(epoch))

    # switch to train mode
    model.train()

    end = time.time()
    for i, (images, target) in enumerate(train_loader):
        lr = adjust_learning_rate(optimizer=optimizer, epoch=epoch, step=i, len_epoch=len(train_loader))
        LR.update(lr)
        # measure data loading time
        data_time.update(time.time() - end)

        if args.gpu is not None:
            images = images.cuda(args.gpu, non_blocking=True)
        if torch.cuda.is_available():
            target = target.cuda(args.gpu, non_blocking=True)

        # compute output
        output = model(images)
        if isinstance(criterion, nn.MSELoss):
            target_onehot = F.one_hot(target, args.num_classes).float()
        else:
            target_onehot = target
        loss = criterion(output, target_onehot)

        # measure accuracy and record loss
        acc1, acc5 = accuracy(output, target, topk=(1, 5))
        losses.update(loss.item(), images.size(0))
        top1.update(acc1[0], images.size(0))
        top5.update(acc5[0], images.size(0))

        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % args.print_freq == 0:
            progress.display(i)


def save_checkpoint(state, is_best,task_id, filename='checkpoint.pth.tar'):
    torch.save(state, os.path.join(args.dirname, str(task_id) + filename))
    if is_best:
        print('New best model')
        shutil.copyfile(os.path.join(args.dirname, str(task_id) + filename), os.path.join(args.dirname, str(task_id) + 'model_best.pth.tar'))

class Summary(Enum):
    NONE = 0
    AVERAGE = 1
    SUM = 2
    COUNT = 3


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self, name, fmt=':f', summary_type=Summary.AVERAGE):
        self.name = name
        self.fmt = fmt
        self.summary_type = summary_type
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def __str__(self):
        fmtstr = '{name} {val' + self.fmt + '} ({avg' + self.fmt + '})'
        return fmtstr.format(**self.__dict__)

    def summary(self):
        fmtstr = ''
        if self.summary_type is Summary.NONE:
            fmtstr = ''
        elif self.summary_type is Summary.AVERAGE:
            fmtstr = '{name} {avg:.3f}'
        elif self.summary_type is Summary.SUM:
            fmtstr = '{name} {sum:.3f}'
        elif self.summary_type is Summary.COUNT:
            fmtstr = '{name} {count:.3f}'
        else:
            raise ValueError('invalid summary type %r' % self.summary_type)

        return fmtstr.format(**self.__dict__)


class ProgressMeter(object):
    def __init__(self, num_batches, meters, prefix=""):
        self.batch_fmtstr = self._get_batch_fmtstr(num_batches)
        self.meters = meters
        self.prefix = prefix

    def display(self, batch):
        entries = [self.prefix + self.batch_fmtstr.format(batch)]
        entries += [str(meter) for meter in self.meters]
        print('\t'.join(entries))

    def display_summary(self):
        entries = [" *"]
        entries += [meter.summary() for meter in self.meters]
        print(' '.join(entries))

    def _get_batch_fmtstr(self, num_batches):
        num_digits = len(str(num_batches // 1))
        fmt = '{:' + str(num_digits) + 'd}'
        return '[' + fmt + '/' + fmt.format(num_batches) + ']'

def adjust_learning_rate(optimizer, epoch, step, len_epoch):
    """LR schedule that should yield 76% converged accuracy with batch size 256"""
    for i in range(len(args.lr_decay_milestones)):
        if epoch >= args.lr_decay_milestones[-1]:
            scaling = len(args.lr_decay_milestones)
            break
        elif epoch < args.lr_decay_milestones[i]:
            scaling = i
            break
    lr = args.lr * 10 ** (-scaling)
    """Warmup"""
    if epoch < args.warm_up_epochs:
        lr = 0.01 * args.lr + (args.lr - 0.01 * args.lr) * (step + 1 + epoch * len_epoch) / (
                    args.warm_up_epochs * len_epoch)
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

    return lr

#这里开始是第一个task上的训练

import resnet_cifar as model_cifar

if args.phase > 0:
    nc_each = (args.full_classes - args.baseclass) // args.phase
else:
    nc_each = 0

# 定义一个memory容器
memory_data = {}

#base_training:

model = model_cifar.__dict__[args.arch](100)

model = model.cuda(args.gpu)

criterion = nn.CrossEntropyLoss().cuda(args.gpu)
optimizer = torch.optim.SGD(model.parameters(), args.lr,
                            momentum=args.momentum,
                            weight_decay=args.weight_decay)

task_id = 0
best_acc1 = 0
for epoch in range(args.start_epoch, args.epochs):
    if args.distributed:
        train_sampler.set_epoch(epoch)
    # adjust_learning_rate(optimizer, epoch, args)

    # train for one epoch
    train(train_loader, model, criterion, optimizer, epoch, args)

    # evaluate on validation set
    acc1 = validate(val_loader, model, criterion, args, print=True)

    # remember best acc@1 and save checkpoint
    is_best = acc1 > best_acc1

    best_acc1 = max(acc1, best_acc1)

    if not args.multiprocessing_distributed or (args.multiprocessing_distributed
                                                and args.rank % ngpus_per_node == 0):
        save_checkpoint({
            'epoch': epoch + 1,
            'arch': args.arch,
            'state_dict': model.state_dict(),
            'best_acc1': best_acc1,
            'optimizer': optimizer.state_dict(),
        }, is_best,task_id)



for phase in range(args.phase):
    task_id = phase + 1
    # train_loader, val_loader = get_cifar100fx0(50+nc_each*phase)
    args.num_classes = args.baseclass + nc_each * (task_id)
    # model.module.fc[-1] = nn.Linear(args.Hidden, args.num_classes, bias=False)
    print('Learning for Phase {}/{}'.format(task_id, args.phase))
    # train_loader = get_IL_dataset(train_loader, IL_dataset_train[phase], True)

    #加载当前task的数据集
    train_loader = IL_dataset_train[phase]
    val_loader = IL_dataset_val[phase]

    # 重新初始化当前训练的网络
    model = model_cifar.__dict__[args.arch](100)

    model = model.cuda(args.gpu)

    criterion = nn.CrossEntropyLoss().cuda(args.gpu)
    optimizer = torch.optim.SGD(model.parameters(), args.lr,
                                momentum=args.momentum,
                                weight_decay=args.weight_decay)
    best_acc1 = 0
    for epoch in range(args.start_epoch, args.epochs):
        if args.distributed:
            train_sampler.set_epoch(epoch)
        # adjust_learning_rate(optimizer, epoch, args)

        # train for one epoch
        train(train_loader, model, criterion, optimizer, epoch, args)

        # evaluate on validation set
        acc1 = validate(val_loader, model, criterion, args, print=True)

        # remember best acc@1 and save checkpoint
        is_best = acc1 > best_acc1

        best_acc1 = max(acc1, best_acc1)

        if not args.multiprocessing_distributed or (args.multiprocessing_distributed
                                                    and args.rank % ngpus_per_node == 0):
            save_checkpoint({
                'epoch': epoch + 1,
                'arch': args.arch,
                'state_dict': model.state_dict(),
                'best_acc1': best_acc1,
                'optimizer': optimizer.state_dict(),
            }, is_best, task_id)


    print('Base phase {}/{}: {}%'.format(task_id, args.phase, acc1))





#
# for count in range(1, 6):  # 五个分类器
#     print(f'step{count}')
#     # Step0：取最新的dataloader
#     # 第一次：1，2批数据进来，训练分类器1
#     # 第二次：3数据进来，训练分类器1，2
#     # 第n次：n+1数据进来，训练分类器1，2，...n
#
#     if count == 1:  # 如果是第一次，那么需要把1，2批数据合并
#         dataloader = ConcatDataset([dataloader.dataset, train_loaders[count].dataset])  # 新增数据集
#         dataloader = DataLoader(dataloader, batch_size=128, shuffle=True)
#     else:  # 不是第一次，就直接用最新的dataloader
#         dataloader = train_loaders[count].dataset
#         dataloader = DataLoader(dataloader, batch_size=128, shuffle=True)
#
#     # Step1：更新memory
#     # 计算当前每类数量
#     cur_class_num = int(memory_size/(50+10*count)) + 1  # 每次进来10个类，每次分母+10，这里+1是让它多存一点（反正memorysize稍微一两个也无所谓）
#     if memory_data != {}:
#         for label_id in memory_data.keys():  # 遍历所有memory中的类别，先将其大小缩小为cur_class_num
#             memory_data[label_id] = random.sample(memory_data[label_id], cur_class_num)
#     for data, labels in dataloader:  # 遍历dataloader中每个数据
#         for i in range(len(labels)):
#             label = labels[i].item()  # 取出标签值
#             # 如果标签不在memory中就新建一个key
#             if label not in memory_data:
#                 memory_data[label] = []
#             # 如果已经有该标签的key，则存入
#             if len(memory_data[label]) < cur_class_num:
#                 memory_data[label].append(data[i])
#     # 完成了memory的更新
#
#     # Step2：开始训练
#     if count != 1:  # 除非是第一次，否则都需要加载memory
#         images = []
#         labels = []
#         for label, image_list in memory_data.items():
#             for image in image_list:
#                 images.append(image)
#                 labels.append(label)
#
#         images = torch.stack(images)
#         labels = torch.tensor(labels)
#
#         dataset_m = TensorDataset(images, labels)
#         dataloader_m = DataLoader(dataset_m, batch_size=128)  # 把memory装在为一个dataloader
#
#     # 对前面几个net训练head
#     if count != 1:
#         for train_nets in range(count-1):  # 对前面几个net训练head
#             model = nets[train_nets]  # 确定当前训练的是哪个网络
#             # 待完成：冻结当前net的backbone
#
#             # 定义当前dataloader，为dataloader_m加上少量当前数据集（暂用更新后的dataloader，这里还没改成先训练后更新memory）
#             dataloader_cur = dataloader_m
#             for epoch in range(epochs):  # 假设我们训练10个epochs
#                 running_loss = 0.0
#                 for i, (inputs, labels) in enumerate(dataloader_cur, 0):
#                     # inputs, labels = data[0].to(device), data[1].to(device)
#                     # 0:50-60
#                     # 1:60-70
#                     # 2:70-80
#                     # 3:80-90
#                     # 4:90-100
#
#                     # labels = torch.tensor([0 if i < 50 else 1 if (50 + train_nets * 10) <= i < (60 + train_nets * 10) else 0 for i in labels]).to(device)
#                     # 对于标签，属于当前类别的视为ind，标签为1，否则ood，标签为0
#                     inputs = inputs.to(device)
#                     labels = labels.to(device)
#                     # 将优化器的梯度归零
#                     optimizer.zero_grad()
#
#                     # 前向传播，计算损失，反向传播，优化
#                     outputs = model(inputs)
#                     loss = criterion(outputs, labels)
#                     # print(loss)
#                     loss.backward()
#                     optimizer.step()
#
#                     running_loss += loss.item()
#                 print(f"Epoch {epoch + 1}, loss: {running_loss / len(dataloader)}")
#             nets[train_nets] = model  # 更新网络
#
#     # 对当前net训练full para
#     model = nets[count - 1]
#     for epoch in range(epochs):  # 假设我们训练10个epochs
#         running_loss = 0.0
#         for i, (inputs, labels) in enumerate(dataloader, 0):
#             # inputs, labels = data[0].to(device), data[1].to(device)
#             # 0:50-60
#             # 1:60-70
#             # 2:70-80
#             # 3:80-90
#             # 4:90-100
#
#             # labels = torch.tensor([0 if i < 50 else 1 if (50 + train_nets * 10) <= i < (60 + train_nets * 10) else 0 for i in labels]).to(device)
#             # 对于标签，属于当前类别的视为ind，标签为1，否则ood，标签为0
#             inputs = inputs.to(device)
#             labels = labels.to(device)
#             # 将优化器的梯度归零
#             optimizer.zero_grad()
#
#             # 前向传播，计算损失，反向传播，优化
#             outputs = model(inputs)
#             loss = criterion(outputs, labels)
#             # print(loss)
#             loss.backward()
#             optimizer.step()
#
#             running_loss += loss.item()
#         print(f"Epoch {epoch + 1}, loss: {running_loss / len(dataloader)}")
#     nets[count - 1] = model  # 更新网络
#
# for i in range(0, len(nets)):
#     torch.save(nets[i].state_dict(), f'model{i}_param.pth')
print('Finished Training')
