import math
import os
import shutil

import cv2
import imageio
import numpy as np
import torch
import torch.optim as optim
from skimage import filters
from torch.autograd import Variable


def save_checkpoint(
    state, is_best, fold, savename, epoch, filename="model_checkpoint.pth.tar"
):
    dirname = "{}".format(savename)
    torch.save(state, os.path.join(dirname, filename))
    if is_best:
        print("Saving checkpoint {} as the best model...".format(epoch))
        shutil.copyfile(
            os.path.join(dirname, filename),
            "{}/model_best_{}.pth.tar".format(savename, str(fold)),
        )


def adjust_learning_rate(optimizer, epoch, epochs, lr, cos=True, schedule=None):
    """Decay the learning rate based on schedule"""
    if cos:  # cosine lr schedule
        lr *= 0.5 * (1.0 + math.cos(math.pi * epoch / epochs))
    else:  # stepwise lr schedule
        for milestone in schedule:
            lr *= 0.1 if epoch >= milestone else 1.0
    for param_group in optimizer.param_groups:
        param_group["lr"] = lr


def adjust_alpha(epoch, epochs, min_alpha=0.2):
    step = (1 - min_alpha) / epochs
    return 1 - epoch * step


def make_optimizer(args):
    """make optimizer"""
    # optimizer
    kwargs_optimizer = {"lr": args.lr}

    if args.optimizer == "SGD":
        optimizer_class = optim.SGD
        kwargs_optimizer["momentum"] = args.momentum
    elif args.optimizer == "ADAM":
        optimizer_class = optim.Adam
        kwargs_optimizer["betas"] = (0.9, 0.999)
        kwargs_optimizer["eps"] = 1e-8
    elif args.optimizer == "RMSprop":
        optimizer_class = optim.RMSprop
        kwargs_optimizer["eps"] = args.epsilon
    else:
        optimizer_class = optim.Adam
        kwargs_optimizer["betas"] = (0.9, 0.999)
        kwargs_optimizer["eps"] = 1e-8

    return optimizer_class, kwargs_optimizer


def get_rank() -> int:
    # SLURM_PROCID can be set even if SLURM is not managing the multiprocessing,
    # therefore LOCAL_RANK needs to be checked first
    rank_keys = ("RANK", "LOCAL_RANK", "SLURM_PROCID")
    for key in rank_keys:
        rank = os.environ.get(key)
        if rank is not None:
            return int(rank)
    return 0


def get_soft_label(input_tensor, num_class):
    """
    convert a label tensor to soft label
    input_tensor: tensor with shape [N, C, H, W]
    output_tensor: shape [N, H, W, num_class]
    """
    tensor_list = []
    input_tensor = input_tensor.permute(0, 2, 3, 1)
    for i in range(num_class):
        temp_prob = torch.eq(input_tensor, i * torch.ones_like(input_tensor))
        tensor_list.append(temp_prob)
    output_tensor = torch.cat(tensor_list, dim=-1)
    output_tensor = output_tensor.float()
    return output_tensor
