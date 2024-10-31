import os
import random
import warnings
import argparse

import torch

import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.distributed as dist
import torch.optim
import torch.multiprocessing as mp
import torch.utils.data
import torch.utils.data.distributed
import torchvision.transforms as transforms
import torchvision.models as models
import torchvision.datasets as datasets
from torch.optim.lr_scheduler import LambdaLR
import math
import time
import logging
from tqdm import tqdm
import torch.nn as nn
import torch.nn.functional as F
import sys
sys.path.append('../')
from EDC_validation.utils import (
    ShufflePatches,
    mix_aug,
    AverageMeter,
    accuracy,
    get_parameters,
    load_model
)
from copy import deepcopy
from data import ImageFolder
from misc.utils import Logger, Plotter

class EMAMODEL(object):
    def __init__(self,model):
        self.ema_model = deepcopy(model)
        for parameter in self.ema_model.parameters():
            parameter.requires_grad_(False)
        self.ema_model.eval()

    @torch.no_grad()
    def ema_step(self,decay_rate=0.999,model=None):
        for param,ema_param in zip(model.parameters(),self.ema_model.parameters()):
            ema_param.data.mul_(decay_rate).add_(param.data, alpha=1. - decay_rate)
    
    @torch.no_grad()
    def ema_swap(self,model=None):
        # print('Begin swap',list(self.ema_model.parameters())[0].data[0,0],list(model.parameters())[0].data[0,0])
        for param,ema_param in zip(self.ema_model.parameters(),model.parameters()):
            tmp = param.data.detach()
            param.data = ema_param.detach()
            ema_param.data = tmp
        # print('After swap',list(self.ema_model.parameters())[0].data[0,0],list(model.parameters())[0].data[0,0])
    
    @torch.no_grad()
    def __call__(self, pre_z_t,t):
        return self.ema_model.module(pre_z_t,t)

sharing_strategy = "file_system"
torch.multiprocessing.set_sharing_strategy(sharing_strategy)


def set_worker_sharing_strategy(worker_id: int) -> None:
    torch.multiprocessing.set_sharing_strategy(sharing_strategy)


def main(args, logger, repeat=1):
    if args.seed is not None:
        random.seed(args.seed)
        torch.manual_seed(args.seed)
    cudnn.benchmark = True
    
    best_acc_list = []
    logger(f"ImageNet directory: {args.syn_data_path}")
    for i in range(repeat):
        logger(f"Repeat {i+1}")
        # plotter = Plotter(args.log_path, args.re_epochs, idx=i)
        best_acc = main_worker(args, logger)
        best_acc_list.append(best_acc)
    
    import numpy as np
    logger(f'\n(Repeat {repeat}) Best, last acc: {np.mean(best_acc_list):.1f} {np.std(best_acc_list):.1f}')

def main_worker(args, logger):
    logger("=> using pytorch pre-trained teacher model '{}'".format(args.arch_name))
    # TODO: ensemble
    # teacher_model = load_model(
    #     model_name=args.arch_name,
    #     dataset=args.subset,
    #     pretrained=True,
    #     classes=args.classes,
    # )
    aux_teacher = ["resnet18", "mobilenet_v2", "efficientnet_b0", "shufflenet_v2_x0_5"]
    # aux_teacher = ["resnet18"]
    args.pre_train_path = '/root/workspace/MinimaxDiffusion/squeeze/squeeze_wo_ema'
    teacher_models = []
    for name in aux_teacher:    
        model_t = models.__dict__[name](pretrained=False, num_classes=10)
        teacher_models.append(model_t)
        checkpoint = torch.load(
            os.path.join(args.pre_train_path, args.spec, name, f"squeeze_{name}.pth"),
            map_location="cpu")
        teacher_models[-1].load_state_dict(checkpoint)
        
    for _model in teacher_models:
        _model.cuda()
        _model = torch.nn.DataParallel(_model)
        _model.eval()
        for param in _model.parameters():
            param.requires_grad = False

    student_model = load_model(
        model_name=args.stud_name,
        dataset=args.subset,
        pretrained=False,
        classes=args.classes,
    )
    # FIXME ensemble
    # teacher_model = teacher_models[0]
    # teacher_model = torch.nn.DataParallel(teacher_model).cuda()
    student_model = torch.nn.DataParallel(student_model).cuda()

    # teacher_model.eval()
    student_model.train()
    ema_student_model = EMAMODEL(student_model)

    # freeze all layers
    # for param in teacher_model.parameters():
    #     param.requires_grad = False

    cudnn.benchmark = True

    # optimizer
    if args.sgd:
        optimizer = torch.optim.SGD(
            get_parameters(student_model),
            lr=args.learning_rate,
            momentum=args.momentum,
            weight_decay=args.weight_decay,
        )
    else:
        optimizer = torch.optim.AdamW(
            get_parameters(student_model),
            lr=args.adamw_lr,
            betas=[0.9, 0.999],
            weight_decay=args.adamw_weight_decay,
        )

    # lr scheduler
    if args.cos == True:
    #     scheduler = LambdaLR(
    #         optimizer,
    #         lambda step: 0.5 * (1.0 + math.cos(math.pi * step / args.re_epochs / 2))
    #         if step <= args.re_epochs
    #         else 0,
    #         last_epoch=-1,
    #     )
    # elif args.ls_type == "cos2":
        scheduler = LambdaLR(optimizer,
                             lambda step: 0.5 * (
                                     1. + math.cos(math.pi * step / (2 * args.re_epochs))) if step <= (args.re_epochs*5/6) else 
                                     0.5 * (1. + math.cos(math.pi * 5 / (6 * 2))) * (6*args.re_epochs-6*step)/(6*args.re_epochs),
                             last_epoch=-1)
    
    else:
        scheduler = LambdaLR(
            optimizer,
            lambda step: (1.0 - step / args.re_epochs) if step <= args.re_epochs else 0,
            last_epoch=-1,
        )

    logger("process data from {}".format(args.syn_data_path))
    normalize = transforms.Normalize(
        mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
    )

    augment = []
    augment.append(transforms.RandAugment())
    augment.append(transforms.ToTensor())
    augment.append(ShufflePatches(args.factor))
    augment.append(
        transforms.RandomResizedCrop(
            size=args.input_size,
            scale=(1 / args.factor, args.max_scale_crops),
            antialias=True,
        )
    )
    augment.append(transforms.RandomHorizontalFlip())
    augment.append(normalize)
    # change to ImageNetwithClass
    # train_dataset = ImageFolder(
    #     classes=range(args.nclass),
    #     ipc=args.ipc,
    #     mem=True,
    #     shuffle=True,
    #     root=args.syn_data_path,
    #     transform=transforms.Compose(augment),
    # )
    train_dataset = ImageFolder(args.syn_data_path,
                                transform=transforms.Compose(augment),
                                nclass=args.nclass,
                                slct_type=args.slct_type,
                                ipc=args.ipc,
                                load_memory=args.load_memory,
                                spec=args.spec)

    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=args.re_batch_size,
        shuffle=True,
        num_workers=args.workers,
        pin_memory=True,
        worker_init_fn=set_worker_sharing_strategy,
    )
    
    val_dataset = ImageFolder(root=args.val_dir,
                                transform=transforms.Compose([
                                    transforms.Resize(256),
                                    transforms.CenterCrop(224),
                                    transforms.ToTensor(),
                                    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                                        std=[0.229, 0.224, 0.225]),
                                ]),
                            nclass=args.nclass,
                            load_memory=args.load_memory,
                            spec=args.spec)
    
    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=args.re_batch_size,
        shuffle=False,
        num_workers=args.workers,
        pin_memory=True,
        worker_init_fn=set_worker_sharing_strategy,
    )
    logger("load data successfully")

    best_acc1 = 0
    best_epoch = 0
    args.optimizer = optimizer
    args.scheduler = scheduler
    args.train_loader = train_loader
    args.val_loader = val_loader

    for epoch in range(args.re_epochs):
        train(epoch, train_loader, teacher_models, student_model, args, ema_model=ema_student_model, logger=logger)

        if epoch % 10 == 9 or epoch == args.re_epochs - 1:
            if epoch > args.re_epochs * 0.8:
                ema_student_model.ema_swap(student_model)
                top1 = validate(student_model, args, epoch)
                ema_student_model.ema_swap(student_model)
            else:
                top1 = 0
        else:
            top1 = 0

        scheduler.step()
        if top1 > best_acc1:
            best_acc1 = max(top1, best_acc1)
            best_epoch = epoch

    logger(f"Train Finish! Best accuracy is {best_acc1}@{best_epoch}")
    return best_acc1



def train(epoch, train_loader, teacher_models, student_model, args, ema_model=None, logger=None):
    """Generate soft labels and train"""
    objs = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()

    optimizer = args.optimizer
    loss_function_kl = nn.KLDivLoss(reduction="batchmean")
    # teacher_model.eval()
    student_model.train()
    t1 = time.time()
    for batch_idx, (images, labels) in enumerate(train_loader):
        with torch.no_grad():
            images = images.cuda()
            labels = labels.cuda()

            mix_images, _, _, _ = mix_aug(images, args)

            pred_label = student_model(images)

            # soft_mix_label = teacher_model(mix_images)
            # soft_mix_label = F.softmax(soft_mix_label / args.temperature, dim=1)
            with torch.no_grad():
                soft_label = []
                for _model in teacher_models:
                    soft_label.append(_model(mix_images))
                
                soft_label = torch.stack(soft_label, 0)
                soft_label = soft_label.mean(0)
                soft_label = F.softmax(soft_label / args.temperature, dim=1)


        if batch_idx % args.re_accum_steps == 0:
            optimizer.zero_grad()

        prec1, prec5 = accuracy(pred_label, labels, topk=(1, 5))

        pred_mix_label = student_model(mix_images)

        soft_pred_mix_label = F.log_softmax(pred_mix_label / args.temperature, dim=1)
        loss = loss_function_kl(soft_pred_mix_label, soft_label)

        loss = loss / args.re_accum_steps

        loss.backward()
        if batch_idx % args.re_accum_steps == (args.re_accum_steps - 1):
            optimizer.step()

        if ema_model is not None:
            ema_model.ema_step(model=student_model,decay_rate=0.99)

        n = images.size(0)
        objs.update(loss.item(), n)
        top1.update(prec1.item(), n)
        top5.update(prec5.item(), n)

    printInfo = (
        "TRAIN Iter {}: loss = {:.6f},\t".format(epoch, objs.avg)
        + "Top-1 err = {:.6f},\t".format(100 - top1.avg)
        + "Top-5 err = {:.6f},\t".format(100 - top5.avg)
        + "train_time = {:.6f}".format((time.time() - t1))
    )
    logger(printInfo)
    t1 = time.time()


def validate(model, args, epoch=None):
    objs = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()
    loss_function = nn.CrossEntropyLoss()

    model.eval()
    t1 = time.time()
    with torch.no_grad():
        for data, target in args.val_loader:
            target = target.type(torch.LongTensor)
            data, target = data.cuda(), target.cuda()

            output = model(data)
            loss = loss_function(output, target)

            prec1, prec5 = accuracy(output, target, topk=(1, 5))
            n = data.size(0)
            objs.update(loss.item(), n)
            top1.update(prec1.item(), n)
            top5.update(prec5.item(), n)

    logInfo = (
        "TEST:\nIter {}: loss = {:.6f},\t".format(epoch, objs.avg)
        + "Top-1 err = {:.6f},\t".format(100 - top1.avg)
        + "Top-5 err = {:.6f},\t".format(100 - top5.avg)
        + "val_time = {:.6f}".format(time.time() - t1)
    )
    print(logInfo)
    logInfo = (
        "TEST: Iter {}: loss = {:.6f},\t".format(epoch, objs.avg)
        + "Top-1 err = {:.6f},\t".format(100 - top1.avg)
        + "Top-5 err = {:.6f},\t".format(100 - top5.avg)
        + "val_time = {:.6f}".format(time.time() - t1)
    )
    return top1.avg


if __name__ == "__main__":
    from misc.utils import Logger
    from EDC_validation.valid_argument import args

    logger = Logger(args.log_path)
    main(args, logger, args.repeat)
