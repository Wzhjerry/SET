import glob
import os
import random
import time
from builtins import len

import cv2
import imageio
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from sklearn.model_selection import KFold
from torch.utils.data import Dataset
from torchvision import transforms
from tqdm import tqdm

from datasets.data_utils import analyze_name, load_group_associations

cv2.setNumThreads(1)


class ISIC18(Dataset):
    def __init__(self, x, y, names, im_transform, label_transform, sp_num, size, stage, train=False, test=False):
        self.im_transform = im_transform
        self.label_transform = label_transform
        assert len(x) == len(y)
        assert len(x) == len(names)
        self.dataset_size = len(y)
        self.x = x
        self.y = y
        self.names = names
        self.sp_num = sp_num
        self.size = size
        self.stage = stage
        self.train = train
        self.test = test

    def __len__(self):
        if self.train:
            return self.dataset_size * len(self.stage)
        else:
            return self.dataset_size

    def _get_index(self, idx):
        if self.train:
            return idx % self.dataset_size, idx // self.dataset_size
        else:
            return idx, 0

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        idx, indent = self._get_index(idx)

        # image
        input = cv2.imread(self.x[idx])[..., ::-1]
        input = input[2:-2, 2:-2, :]
        input = cv2.resize(input, (512, 512), interpolation=cv2.INTER_CUBIC)
        # label
        target = cv2.imread(self.y[idx])
        target = cv2.resize(target, (512, 512), interpolation=cv2.INTER_NEAREST)
        target = target[..., 0]
        # Name
        name = self.names[idx]
        # asscoiation matrix
        matrix_sp, matrix_a1, matrix_a2, matrix_a3, matrix_a4 = load_group_associations(
            path=self.x[idx],
            size=self.size,
            sp_num=self.sp_num,
            stage=self.stage if self.test else self.stage[indent],
            pattern='Training_Data'
        )

        im = Image.fromarray(np.uint8(input))
        target = Image.fromarray(np.uint8(target)).convert("1")

        # identical transformation for im and gt
        seed = np.random.randint(2147483647)
        torch.manual_seed(seed)
        random.seed(seed)

        if self.im_transform is not None:
            im_t = self.im_transform(im)

        torch.manual_seed(seed)
        random.seed(seed)
        if self.label_transform is not None:
            target_t = self.label_transform(target)
            target_t = torch.squeeze(target_t).long()

        # import imageio
        # im_np = (im_t.permute(1, 2, 0).numpy() * 0.5 + 0.5) * 255
        # target_np = (target_t.numpy()) * 255
        # imageio.imwrite('./debug/im.png', np.array(im_np).astype(np.uint8))
        # imageio.imwrite('./debug/gt.png', np.array(target_np).astype(np.uint8))

        if self.train:
            return im_t, target_t, matrix_sp, matrix_a1, matrix_a2, matrix_a3, matrix_a4
        else:
            return im_t, target_t, matrix_sp, matrix_a1, matrix_a2, matrix_a3, matrix_a4, name


def load_name():

    inputs, targets, names = [], [], []

    input_pattern = glob.glob(
        "/data2/wangzh/datasets/ISIC2018/Training_Data/*.jpg"
    )
    targetlist = (
        "/data2/wangzh/datasets/ISIC2018/Training_GroundTruth/{}_segmentation.png"
    )

    input_pattern.sort()

    for i in tqdm(range(len(input_pattern))):
        inputpath = input_pattern[i]
        name = analyze_name(inputpath)
        targetpath = targetlist.format(str(name))

        if os.path.exists(inputpath):
            inputs.append(inputpath)
            targets.append(targetpath)
            names.append(name)

    inputs = np.array(inputs)
    targets = np.array(targets)
    names = np.array(names)

    return inputs, targets, names


def load_dataset(args, fold, train=True):
    inputs, targets, names = load_name()
    index = int(args.percent * len(inputs) / 100)
    if args.percent < 1:
        inputs = inputs[444:index + 445]
        targets = targets[444:index + 445]
        names = names[444:index + 445]
    else:
        inputs = inputs[:index]
        targets = targets[:index]
        names = names[:index]
    print("Length of new inputs:", len(inputs))
    # mean & variance
    # mean, std = stat(inputs, masks)
    normalize = transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])

    kf = KFold(n_splits=5, shuffle=True, random_state=436)
    for ifold, (train_index, test_index) in enumerate(kf.split(inputs)):
        test_index = np.append(test_index, train_index[-1])
        if ifold != fold:
            continue
        X_trainset, X_test = inputs[train_index], inputs[test_index]
        y_trainset, y_test = targets[train_index], targets[test_index]
        names_trainset, names_test = names[train_index], names[test_index]

    index_s = int(len(X_trainset) * 0.875)
    X_train, X_val, y_train, y_val, names_train, names_val = (
        X_trainset[:index_s],
        X_trainset[index_s:],
        y_trainset[:index_s],
        y_trainset[index_s:],
        names_trainset[:index_s],
        names_trainset[index_s:],
    )
    print("Length of inputs: {}, {}, {}".format(len(X_train), len(X_val), len(X_test)))

    # transform input images and construct datasets
    size = args.size
    transform = transforms.Compose(
        [
            transforms.Resize(size),
            transforms.ColorJitter(0.2, 0.2, 0.2, 0.2),
            transforms.RandomGrayscale(p=0.2),
            transforms.RandomAutocontrast(p=0.2),
            transforms.RandomInvert(p=0.2),
            transforms.ToTensor(),
            normalize,
        ]
    )
    label_transform = transforms.Compose(
        [
            transforms.Resize(size, interpolation=Image.NEAREST),
            transforms.ToTensor(),
        ]
    )
    test_transform = transforms.Compose(
        [
            transforms.Resize(size),
            transforms.ToTensor(),
            normalize,
        ]
    )
    test_label_transform = transforms.Compose(
        [
            transforms.Resize(size, interpolation=Image.NEAREST),
            transforms.ToTensor(),
        ]
    )

    train_dataset = ISIC18(
        X_train,
        y_train,
        names_train,
        im_transform=transform,
        label_transform=label_transform,
        sp_num=args.sp_num,
        size=args.size,
        stage=args.stage,
        train=True,
    )
    val_dataset = ISIC18(
        X_val,
        y_val,
        names_val,
        im_transform=test_transform,
        label_transform=test_label_transform,
        sp_num=args.sp_num,
        size=args.size,
        stage=args.stage,
        train=False,
        test=args.ensemble,
    )
    test_dataset = ISIC18(
        X_test,
        y_test,
        names_test,
        im_transform=test_transform,
        label_transform=test_label_transform,
        sp_num=args.sp_num,
        size=args.size,
        stage=args.stage,
        train=False,
        test=args.ensemble,
    )

    if train:
        return train_dataset, test_dataset
    else:
        return test_dataset
