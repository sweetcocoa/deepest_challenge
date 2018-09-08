import argparse
import os
import random
import shutil
import time
import warnings
# import PinkBlack.io

from PIL import Image
import numpy as np

import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.distributed as dist
import torch.optim
import torch.utils.data
import torch.utils.data.distributed
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torchvision.models as models

"""
**** For Reproduction *** 
 
default_args의 train, test dataset 위치를 지정해줘야 함 (ImageFolder 구조)

batch_size를 실행 환경에 맞게 적당히 조절 

Best Accuracy 모델 : 
- model : "big_resnet",
- pretrained : "big_resnet.pth.best"

Best Score 모델 : 
- model : "splicedresnet",
- pretrained : "splicedresnet.pth.best"
 

"""
default_args={"CUDA_VISIBLE_DEVICES": "0",
                "train": "/data/jongho/data/UCSD/CUB_200_2011/images_split/train/",
                "test": "/data/jongho/data/UCSD/CUB_200_2011/images_split/test/",
                # "model": "splicedresnet",
                "model": "big_resnet",
                "batch_size": "256",
                "seed": "0",
                "checkpoint": "big_resnet.pth",
                # "pretrained": "splicedresnet.pth.best",
                "pretrained": "big_resnet.pth.best",
                "epoch": "0"
}
os.environ.update(default_args)

seed = int(os.environ['seed'])
torch.manual_seed(seed)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
np.random.seed(seed)
random.seed(seed)

device = torch.device(0)
best_prec1 = 0

def main():
    global best_prec1

    if os.environ['model'] == "big_resnet":
        model = models.resnet18(pretrained=True, num_classes=1000)
        # model.layer4 = nn.Sequential()
        model.fc = nn.Linear(512, 200)
        model.avgpool = nn.AdaptiveAvgPool2d(1)
    elif os.environ['model'] == "splicedresnet":
        model = models.resnet18(pretrained=True, num_classes=1000)
        model.layer4 = nn.Sequential()
        model.fc = nn.Linear(256, 200)
        model.avgpool = nn.AdaptiveAvgPool2d(1)
    else:
        print("적당한 Model configuration이 아닙니다")
        exit()

    # model = models.vgg11(pretrained=True, num_classes=1000)
    # model.classifier._modules['0'] = nn.Linear(512 * 8 * 8, 4096)
    # model.classifier._modules['6'] = nn.Linear(4096,200)

    # model = models.squeezenet1_1(num_classes=1000, pretrained=True)
    #
    # model.classifier = nn.Sequential(
    #     nn.Linear(512*15*15, 512),
    #     nn.ReLU(inplace=True),
    #     nn.Linear(512, 200),
    # )
    # model.forward = lambda x: model.classifier(model.features(x).view(x.size(0), 512*15*15))

    # last_conv = nn.Conv2d(512, 200, kernel_size=(1, 1), stride=(1, 1))
    # model.num_classes = 200
    # classifier = nn.Sequential(nn.Dropout(p=0.5),last_conv,nn.ReLU(inplace=True),nn.AdaptiveAvgPool2d(1))
    # model.classifier = classifier

    model = model.to(device)

    # define loss function (criterion) and optimizer
    criterion = nn.CrossEntropyLoss().to(device)

    optimizer = torch.optim.Adam(model.parameters())
    # optimizer = torch.optim.SGD(model.parameters(), 0.01,
    #                             momentum=0.9,
    #                             weight_decay=1e-4)
    # optionally resume from a checkpoint

    start_epoch = 0

    if "pretrained" in os.environ.keys() and os.path.exists(os.environ['pretrained']):
        print("=> loading checkpoint '{}'".format(os.environ['pretrained']))
        checkpoint = torch.load(os.environ['pretrained'])
        start_epoch = checkpoint['epoch']
        best_prec1 = checkpoint['best_prec1']
        model.load_state_dict(checkpoint['state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        print("=> loaded checkpoint '{}' (epoch {})"
              .format(os.environ['pretrained'], checkpoint['epoch']))

        def get_n_params(model):
            return sum(p.numel() for p in model.parameters())

        print("=> # of params of model ({})"
              .format(get_n_params(model)))

    cudnn.benchmark = True

    # Data loading code
    traindir = os.environ['train']
    valdir = os.environ['test']
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])

    train_dataset = datasets.ImageFolder(
        traindir,
        transforms.Compose([
            # transforms.RandomResizedCrop(256),
            transforms.Resize((256, 256)),
            transforms.ColorJitter(hue=.05, saturation=.05),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(40, resample=Image.BILINEAR),
            transforms.ToTensor(),
            # normalize,
        ])
    )

    train_sampler = None
    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=int(os.environ['batch_size']), shuffle=(train_sampler is None),
        num_workers=32, pin_memory=True, sampler=train_sampler)

    valid_dataset = datasets.ImageFolder(valdir, transforms.Compose([
            transforms.Resize((256,256)),
            transforms.ToTensor(),
            # normalize,
        ]))
    val_loader = torch.utils.data.DataLoader(
        valid_dataset, batch_size=int(os.environ['batch_size']), shuffle=False,
        num_workers=32, pin_memory=True)

    validate(val_loader, model, criterion)

    for epoch in range(start_epoch, int(os.environ['epoch'])):
        # adjust_learning_rate(optimizer, epoch)

        # train for one epoch

        train(train_loader, model, criterion, optimizer, epoch)

        # evaluate on validation set
        prec1 = validate(val_loader, model, criterion)

        # remember best prec@1 and save checkpoint
        is_best = prec1 > best_prec1
        best_prec1 = max(prec1, best_prec1)
        save_checkpoint({
            'epoch': epoch + 1,
            'arch': os.environ['model'],
            'state_dict': model.state_dict(),
            'best_prec1': best_prec1,
            'optimizer' : optimizer.state_dict(),
        }, is_best)


def train(train_loader, model, criterion, optimizer, epoch):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()

    # switch to train mode
    model.train()

    end = time.time()
    for i, (input, target) in enumerate(train_loader):
        # measure data loading time
        data_time.update(time.time() - end)

        input = input.to(device)
        target = target.to(device)

        # 1/0
        # compute output
        output = model(input)
        loss = criterion(output, target)

        # measure accuracy and record loss
        prec1, prec5 = accuracy(output, target, topk=(1, 5))
        losses.update(loss.item(), input.size(0))
        top1.update(prec1[0], input.size(0))
        top5.update(prec5[0], input.size(0))

        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % 30 == 0:
            print('Epoch: [{0}][{1}/{2}]\t'
                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                  'Prec@1 {top1.val:.3f} ({top1.avg:.3f})\t'
                  'Prec@5 {top5.val:.3f} ({top5.avg:.3f})'.format(
                   epoch, i, len(train_loader), batch_time=batch_time,
                   data_time=data_time, loss=losses, top1=top1, top5=top5))


def validate(val_loader, model, criterion):
    batch_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()

    # switch to evaluate mode
    model.eval()

    with torch.no_grad():
        end = time.time()
        for i, (input, target) in enumerate(val_loader):
            input = input.to(device)
            target = target.to(device)

            # compute output
            output = model(input)
            loss = criterion(output, target)

            # measure accuracy and record loss
            prec1, prec5 = accuracy(output, target, topk=(1, 5))
            losses.update(loss.item(), input.size(0))
            top1.update(prec1[0], input.size(0))
            top5.update(prec5[0], input.size(0))

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            if i % 30 == 0:
                print('Test: [{0}/{1}]\t'
                      'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                      'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                      'Prec@1 {top1.val:.3f} ({top1.avg:.3f})\t'
                      'Prec@5 {top5.val:.3f} ({top5.avg:.3f})'.format(
                       i, len(val_loader), batch_time=batch_time, loss=losses,
                       top1=top1, top5=top5))

        print(' * Prec@1 {top1.avg:.3f} Prec@5 {top5.avg:.3f}'
              .format(top1=top1, top5=top5))

    return top1.avg


def save_checkpoint(state, is_best, filename=os.environ['checkpoint']):
    torch.save(state, filename)
    if is_best:
        shutil.copyfile(filename, filename + ".best")


class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
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

def adjust_learning_rate(optimizer, epoch):
    """Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""
    lr = 0.1 * (0.1 ** (epoch // 30))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

def accuracy(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].view(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res


if __name__ == '__main__':
    main()


