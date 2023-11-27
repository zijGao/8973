# Anonymous Codes of CVPR Paper ID 8973

dependece:
python 3.9.0
pytorch 1.12.0
torchvision 0.13.0

train the model without self-augmentation:

python main.py -a resnet18 --dataset cifar100 --epochs 200 --basetraining --incremental --phase 5 --baseclass 50 --Hidden 10000 --gpu 0

python main.py --ssl -a resnet18 --dataset imagenet100 --epochs 200 --basetraining --incremental --phase 5 --baseclass 50 --Hidden 15000 --gpu 0

python main.py --ssl -a resnet18 --dataset tiny-imagenet --epochs 200 --basetraining --incremental --phase 5 --baseclass 100 --Hidden 15000 --gpu 0

train the model with self-augmentation:

python main_SSLA.py -a resnet18 --dataset cifar100 --epochs 200 --basetraining --incremental --phase 5 --baseclass 50 --Hidden 10000 --gpu 0

python main_SSLA.py --ssl -a resnet18 --dataset imagenet100 --epochs 200 --basetraining --incremental --phase 5 --baseclass 50 --Hidden 15000 --gpu 0

python main_SSLA.py --ssl -a resnet18 --dataset tiny-imagenet --epochs 200 --basetraining --incremental --phase 5 --baseclass 100 --Hidden 15000 --gpu 0
