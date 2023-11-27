import argparse
import torchvision.models as models


parser = argparse.ArgumentParser(description='PyTorch ImageNet Training')

#SSL setting
parser.add_argument('--ssl', action='store_true',
                    help='add SSL')

#KD setting
parser.add_argument('--kd', default=1, type=int, metavar='N',
                    help='the kd weight')
parser.add_argument('--gamma', default=0.6, type=float, metavar='N',
                    help='the ds weight')
#save model
parser.add_argument('--save', action='store_true',
                    help='save model')

#incre_SSL setting
parser.add_argument('--incre_epochs', default=50, type=int, metavar='N',
                    help='number of total epochs to run')


parser.add_argument('--data', metavar='DIR', type=str,
                    help='path to dataset')
parser.add_argument('-a', '--arch', metavar='ARCH', default='resnet18',
                    help='')
parser.add_argument('-d', '--dataset', metavar='DATASET', default='imagenet',
                    help='')
parser.add_argument('-j', '--workers', default=4, type=int, metavar='N',
                    help='number of data loading workers (default: 4)')
parser.add_argument('--seed', default=None, type=int,
                    help='seed for initializing training. ')

######### fundamental parameters
parser.add_argument('--epochs', default=100, type=int, metavar='N',
                    help='number of total epochs to run')
parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                    help='manual epoch number (useful on restarts)')
parser.add_argument('-b', '--batch-size', default=256, type=int,
                    metavar='N',
                    help='mini-batch size (default: 256), this is the total '
                         'batch size of all GPUs on the current node when '
                         'using Data Parallel or Distributed Data Parallel')
parser.add_argument('--lr', '--learning-rate', default=0.1, type=float,
                    metavar='LR', help='initial learning rate', dest='lr')
parser.add_argument('--lr-decay-milestones', nargs='+', type=int, default=[50,75,90],
                    help='decay learning rate at these milestone epochs (default: [200,300,350,375])')
parser.add_argument('--warm-up-epochs', type=int, default=0,
                    help='warming up epochs')
parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                    help='momentum')
parser.add_argument('--wd', '--weight-decay', default=1e-4, type=float,
                    metavar='W', help='weight decay (default: 1e-4)',
                    dest='weight_decay')

######### class incremental learning parameters
parser.add_argument('--subset', action='store_true',
                    help='')
parser.add_argument('--cpnet', action='store_true',
                    help='')
parser.add_argument('--recurbase', action='store_true',
                    help='')
parser.add_argument('--shot', default=5, type=int,
                    help='')
parser.add_argument('--repeat', default=1, type=int,
                    help='')
parser.add_argument('--basetraining', action='store_true',
                    help='base training')
parser.add_argument('--incremental', dest='incremental', action='store_true',
                    help='whether conduct incremental learning')
parser.add_argument('--baseclass', default=50, type=int,
                    help='number of base classes for training')
parser.add_argument('--phase', default=5, type=int,
                    help='')
parser.add_argument('--rg', default=1e-1, type=float,
                    help='')
parser.add_argument('--Hidden', default=1000, type=int,
                    help='')

########### systems
parser.add_argument('-p', '--print-freq', default=50, type=int,
                    metavar='N', help='print frequency (default: 10)')
parser.add_argument('--resume', default='', type=str, metavar='PATH',
                    help='path to latest checkpoint (default: none)')
parser.add_argument('-e', '--evaluate', dest='evaluate', action='store_true',
                    help='evaluate model on validation set')
parser.add_argument('--pretrained', dest='pretrained', action='store_true',
                    help='use pre-trained model')

########### parallel training parameters
parser.add_argument('--world-size', default=-1, type=int,
                    help='number of nodes for distributed training')
parser.add_argument('--rank', default=-1, type=int,
                    help='node rank for distributed training')
parser.add_argument('--dist-url', default='tcp://224.66.41.62:23456', type=str,
                    help='url used to set up distributed training')
parser.add_argument('--dist-backend', default='nccl', type=str,
                    help='distributed backend')
parser.add_argument('--gpu', default=None, type=int,
                    help='GPU id to use.')
parser.add_argument('--multiprocessing-distributed', action='store_true',
                    help='Use multi-processing distributed training to launch '
                         'N processes per node, which has N GPUs. This is the '
                         'fastest way to use PyTorch for either single node or '
                         'multi node data parallel training')

########### others
parser.add_argument('--comt', default='', type=str,
                    help='Comments')

args = parser.parse_args()
args.distributed = args.world_size > 1 or args.multiprocessing_distributed

