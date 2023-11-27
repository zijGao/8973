import copy
import os
import random
import shutil
import time
import warnings
from enum import Enum

from parsers import args
from ssl_utils_ssl import _compute_means,_build_feature_set,expand_Leanrable_model_fc,leanrable_model,_compute_relations,new_model
from resnet_wide import BasicBlock

import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.distributed as dist
import torch.optim
import torch.nn.functional as F
import torch.multiprocessing as mp
import torch.utils.data
import torch.utils.data.distributed
import torchvision.models as models
import numpy as np
import tqdm
best_acc1 = 0

class Flatten(nn.Module):
    def forward(self, x):
        return x.view(x.size(0), -1)


def main():
    print(args)
    if args.resume:
        print(args.resume)
        print('loading model saving directory!')
        if 'model_best.pth.tar' in args.resume:
            dirname = args.resume.replace('/model_best.pth.tar', '')
        elif 'checkpoint.pth.tar' in args.resume:
            dirname = args.resume.replace('/checkpoint.pth.tar', '')

        if 'base_model_best.pth.tar' in args.resume:
            dirname = args.resume.replace('/base_model_best.pth.tar', '')
        args.dirname = dirname
        print(args.dirname)
    else:
        print('creating model saving dir!')
        dirname = ''
        dirname = os.path.join(dirname, 'save_model')
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

    if args.seed is not None:
        random.seed(args.seed)
        torch.manual_seed(args.seed)
        cudnn.deterministic = True
        warnings.warn('You have chosen to seed training. '
                      'This will turn on the CUDNN deterministic setting, '
                      'which can slow down your training considerably! '
                      'You may see unexpected behavior when restarting '
                      'from checkpoints.')

    if args.gpu is not None:
        warnings.warn('You have chosen a specific GPU. This will completely '
                      'disable data parallelism.')

    if args.dist_url == "env://" and args.world_size == -1:
        args.world_size = int(os.environ["WORLD_SIZE"])

    #args.distributed = args.world_size > 1 or args.multiprocessing_distributed

    ngpus_per_node = torch.cuda.device_count()
    if args.multiprocessing_distributed:
        # Since we have ngpus_per_node processes per node, the total world_size
        # needs to be adjusted accordingly
        args.world_size = ngpus_per_node * args.world_size
        # Use torch.multiprocessing.spawn to launch distributed processes: the
        # main_worker process function
        mp.spawn(main_worker, nprocs=ngpus_per_node, args=(ngpus_per_node, args))
    else:
        # Simply call main_worker function
        main_worker(args.gpu, ngpus_per_node, args)


def main_worker(gpu, ngpus_per_node, args):
    global best_acc1
    args.gpu = gpu

    if args.gpu is not None:
        print("Use GPU: {} for training".format(args.gpu))

    from datasets import train_loader, val_loader, train_sampler, num_classes

    args.full_classes = num_classes

    if args.basetraining or args.incremental:
        #from datasets import train_loader_base, val_loader_base, train_sampler, IL_dataset_train, IL_dataset_val, num_classes
        from datasets import get_IL_loader
        train_loader_base, val_loader_base, \
        IL_dataset_train, IL_dataset_val, = get_IL_loader(train_loader.dataset, val_loader.dataset, num_classes)
        num_classes = args.baseclass
        train_loader, val_loader = train_loader_base, val_loader_base
    '''
        train_loader, val_loader = train_loader_base, val_loader_base
    else:
        from datasets import train_loader, val_loader, train_sampler, num_classes
    '''

    if args.distributed:
        if args.dist_url == "env://" and args.rank == -1:
            args.rank = int(os.environ["RANK"])
        if args.multiprocessing_distributed:
            # For multiprocessing distributed training, rank needs to be the
            # global rank among all the processes
            args.rank = args.rank * ngpus_per_node + gpu
        dist.init_process_group(backend=args.dist_backend, init_method=args.dist_url,
                                world_size=args.world_size, rank=args.rank)
    # create model
    if args.pretrained:
        print("=> using pre-trained model '{}'".format(args.arch))
        model = models.__dict__[args.arch](pretrained=True)
    else:
        print("=> creating model '{}'".format(args.arch))
        if args.dataset in ['imagenet100','tiny-imagenet']:
            model = models.__dict__[args.arch](num_classes * 4)
            model.fc = nn.Linear(in_features=512, out_features=num_classes * 4, bias=True)
            args.num_classes = num_classes
        elif args.dataset in ['imagenet']:
            model = models.__dict__[args.arch](num_classes * 4)
            model.fc = nn.Linear(in_features=512, out_features=num_classes * 4, bias=True)
            args.num_classes = num_classes
        elif args.dataset in ['cifar10', 'cifar100']:
            if args.arch in ['resnet18', 'resnet34', 'resnet50', 'resnet101', 'resnet152']:
                import resnet_wide as model_cifar
            else:
                import resnet_cifar as model_cifar
            #from datasets import num_classes
            args.num_classes = num_classes
            model = model_cifar.__dict__[args.arch](args.num_classes * 4 )
        else:
            print('Invalid model!')

    if not torch.cuda.is_available():
        print('using CPU, this will be slow')
    elif args.distributed:
        if args.gpu is not None:
            torch.cuda.set_device(args.gpu)
            model.cuda(args.gpu)
            args.batch_size = int(args.batch_size / ngpus_per_node)
            args.workers = int((args.workers + ngpus_per_node - 1) / ngpus_per_node)
            model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu])
        else:
            model.cuda()
            model = torch.nn.parallel.DistributedDataParallel(model)
    elif args.gpu is not None:
        torch.cuda.set_device(args.gpu)
        model = model.cuda(args.gpu)
    else:
        # DataParallel will divide and allocate batch_size to all available GPUs
        if args.arch.startswith('alexnet') or args.arch.startswith('vgg'):
            model.features = torch.nn.DataParallel(model.features)
            model.cuda()
        else:
            model = torch.nn.DataParallel(model).cuda()

    #create learnable projector and classifier

    model.projector = torch.nn.Sequential(BasicBlock(512,512,1))

    model = new_model(model.conv1, model.bn1, model.relu, model.layer1,
                                                 model.layer2,
                                                 model.layer3, model.layer4, model.projector, model.avgpool, Flatten(), model.fc)

    model = model.cuda(args.gpu)

    model.train()

    # define loss function (criterion) and optimizer
    criterion = nn.CrossEntropyLoss().cuda(args.gpu)
    optimizer = torch.optim.SGD(model.parameters(), args.lr,
                                momentum=args.momentum,
                                weight_decay=args.weight_decay)

 # optionally resume from a checkpoint
    if args.resume:
        if os.path.isfile(args.resume):
            print("=> loading checkpoint '{}'".format(args.resume))
            if args.gpu is None:
                checkpoint = torch.load(args.resume)
            else:
                # Map model to be loaded to specified single gpu.
                loc = 'cuda:{}'.format(args.gpu)
                checkpoint = torch.load(args.resume, map_location=loc)
            args.start_epoch = checkpoint['epoch']
            best_acc1 = checkpoint['best_acc1']
            if args.gpu is not None:
                # best_acc1 may be from a checkpoint from a different GPU
                best_acc1 = best_acc1.to(args.gpu)
            model.load_state_dict(checkpoint['state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer'])
            print("=> loaded checkpoint '{}' (epoch {})"
                  .format(args.resume, checkpoint['epoch']))
        else:
            print("=> no checkpoint found at '{}'".format(args.resume))

    if args.evaluate:
        validate(val_loader, model, criterion, args, print=True)
        return

    for epoch in range(args.start_epoch, 200):
        if args.distributed:
            train_sampler.set_epoch(epoch)
        #adjust_learning_rate(optimizer, epoch, args)


        train(train_loader, model, criterion, optimizer, epoch, args)

        # evaluate on validation set
        acc1 = validate(val_loader, model, criterion, args, print=True,train_mode=True)

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
            }, is_best)




def cpnet(wrapped_model, train_loader, args):
    print('Expanding with CPNet...')
    if hasattr(wrapped_model, 'module'):
        model = wrapped_model.module
    else:
        model = wrapped_model
    if hasattr(model, 'layer4'):
        pass
    else:
        model.layer4 = nn.Sequential()
    if hasattr(model, 'maxpool'):
        pass
    else:
        model.maxpool = nn.Sequential()
    new_model = torch.nn.Sequential(model.conv1, model.bn1, model.relu, model.layer1, model.layer2,
                                    model.layer3, model.layer4, model.avgpool, Flatten())
    model.eval()

    W_fe = torch.zeros(model.fc[0].weight.shape[1], args.Hidden).cuda(args.gpu)
    with torch.no_grad():
        for i, (images, target) in enumerate(train_loader):
            if args.gpu is not None:
                images = images.cuda(args.gpu, non_blocking=True)
            output = new_model(images)
            W_fe += output.t() @ torch.randn(output.shape[0], args.Hidden).cuda(args.gpu)
    W_fe = 0.1*W_fe/(abs(W_fe).max())
    print(W_fe)
    model.fc[0].weight = torch.nn.parameter.Parameter(torch.t(W_fe.float()))

def save_checkpoint(state, is_best, filename='checkpoint.pth.tar'):
    torch.save(state, os.path.join(args.dirname, filename))
    if is_best:
        print('New best model')
        shutil.copyfile(os.path.join(args.dirname, filename), os.path.join(args.dirname, 'model_best.pth.tar'))

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
    lr = args.lr * 10**(-scaling)
    """Warmup"""
    if epoch < args.warm_up_epochs:
        lr = 0.01*args.lr + (args.lr - 0.01*args.lr)*(step + 1 + epoch*len_epoch)/(args.warm_up_epochs * len_epoch)
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

    return lr


def accuracy(output, target, topk=(1,)):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        # if args.ssl and train_mode and validate_mode:
        #     output = output[::4, ::4]
        #     if train_mode:
        #         target = target[::4]

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res



def train(train_loader, model, criterion, optimizer, epoch, args):
    batch_time = AverageMeter('Time', ':6.3f')
    data_time = AverageMeter('Data', ':6.3f')
    losses = AverageMeter('Loss', ':.4e')
    top1 = AverageMeter('Acc@1', ':6.2f')
    top5 = AverageMeter('Acc@5', ':6.2f')
    LR = AverageMeter('lr', ':6.3f') # todo: val not avg
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

#SSL, modify inputs and targets
        if args.ssl:
            #print(images.size())
            images = torch.stack([torch.rot90(images, k, (2, 3)) for k in range(4)], 1)
            if args.dataset == 'cifar100':
                images = images.view(-1, 3, 32, 32)
            elif args.dataset == 'tiny-imagenet':
                images = images.view(-1, 3, 64, 64)
            elif args.dataset == 'imagenet100' or 'imagenet':
                images = images.view(-1, 3, 224, 224)
                #print(images.size())
            target = torch.stack([target * 4 + k for k in range(4)], 1).view(-1)


        # compute output,cruterion=cross_entroy_loss
        project_feature, output = model(images)
        if isinstance(criterion,nn.MSELoss):
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


def validate(val_loader, model, criterion, args, print=True,train_mode=False):
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
            project_feature, output = model(images)


#SSL, choose the first quater
            if args.ssl and train_mode:
                output = output[:,::4]

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
            file.write(str(round(top1.avg.item(),4)) + '\n')

    return top1.avg


if __name__ == '__main__':
    main()