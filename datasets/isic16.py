import random
import glob
import os
import torch
import cv2
import numpy as np
from torch.utils.data import Dataset
from sklearn.model_selection import KFold
from torchvision import transforms
from PIL import Image
from tqdm import tqdm
from datasets.data_utils import analyze_name, load_group_associations


cv2.setNumThreads(1)


class ISIC16(Dataset):
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

        # BGR -> RGB -> PIL
        input = cv2.imread(self.x[idx])[..., ::-1]
        input = input[2:-2, 2:-2, :]
        input = cv2.resize(input, (512, 512), interpolation=cv2.INTER_CUBIC)
        # label
        target = cv2.imread(self.y[idx])
        target = cv2.resize(target, (512, 512), interpolation=cv2.INTER_NEAREST)
        target = target[..., 0]
        # name
        name = self.names[idx]
        # asscoiation matrix
        matrix_sp, matrix_a1, matrix_a2, matrix_a3, matrix_a4 = load_group_associations(
            path=self.x[idx],
            size=self.size,
            sp_num=self.sp_num,
            stage=self.stage if self.test else self.stage[indent],
            pattern='Training_Data' if not self.test else 'Test_Data'
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

        # For debugging
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
    test_inputs, test_targets, test_names = [], [], []

    # Need modification
    input_pattern = glob.glob(
        "/data2/wangzh/datasets/ISIC2016/Training_Data/*.jpg"
    )
    targetlist = (
        "/data2/wangzh/datasets/ISIC2016/Training_GroundTruth/{}_Segmentation.png"
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

    # Need modification
    test_input_pattern = glob.glob(
        "/data2/wangzh/datasets/ISIC2016/Test_Data/*.jpg"
    )
    test_targetlist = (
        "/data2/wangzh/datasets/ISIC2016/Test_GroundTruth/{}_Segmentation.png"
    )

    test_input_pattern.sort()

    for j in tqdm(range(len(test_input_pattern))):
        test_inputpath = test_input_pattern[j]
        test_name = analyze_name(test_inputpath)
        test_targetpath = test_targetlist.format(str(test_name))

        if os.path.exists(test_inputpath):
            test_inputs.append(test_inputpath)
            test_targets.append(test_targetpath)
            test_names.append(test_name)

    test_inputs = np.array(test_inputs)
    test_targets = np.array(test_targets)
    test_names = np.array(test_names)

    assert len(test_inputs) == len(test_targets)
    assert len(test_targets) == len(test_names)

    return inputs, targets, names, test_inputs, test_targets, test_names


def load_dataset(args, fold, train=True, aug_k=40, aug_n=1, patch=False):
    inputs, targets, names, test_inputs, test_targets, test_names = load_name()
    normalize = transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])

    kf = KFold(n_splits=5, shuffle=True, random_state=726)
    for ifold, (train_index, val_index) in enumerate(kf.split(inputs)):
        if ifold != fold:
            continue
        X_train, X_val = inputs[train_index], inputs[val_index]
        y_train, y_val = targets[train_index], targets[val_index]
        names_train, names_val = names[train_index], names[val_index]
    X_test = test_inputs
    y_test = test_targets
    names_test = test_names

    # transform input images and construct datasets
    size = args.size
    transform = transforms.Compose(
        [
            transforms.Resize(size),
            # transforms.ColorJitter(0.2, 0.2, 0.2, 0.2),
            transforms.RandomGrayscale(p=0.5),
            transforms.RandomAutocontrast(p=0.5),
            transforms.RandomInvert(p=0.5),
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

    train_dataset = ISIC16(
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
    val_dataset = ISIC16(
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
    test_dataset = ISIC16(
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
