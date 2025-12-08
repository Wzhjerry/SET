import os
import cv2
import torch
import numpy as np


def analyze_name(path):
    name = os.path.split(path)[1]
    name = os.path.splitext(name)[0]
    return name


def generate_patch_associations(img_size, size, reformat=False):
    # generate a matrix with size of img_size * img_size
    ind = np.zeros((img_size, img_size), dtype=np.int32)
    cnt = 0
    for i in range(0, img_size, size):
        for j in range(0, img_size, size):
            ind[i:i + size, j:j + size] = cnt
            cnt += 1
    ind = ind.flatten()
    if reformat:
        encoded_ind = np.zeros((img_size * img_size, cnt), dtype=np.int32)
        encoded_ind[np.arange(ind.size), ind] = 1

        return encoded_ind
    else:
        return ind


def load_group_associations(path, size, sp_num, stage, pattern):

    if isinstance(stage, list):
        # matrix_sps, matrix_a1s, matrix_a2s, matrix_a3s, matrix_a4s = [], [], [], [], []
        for i in range(len(stage)):
            matrix_sp, matrix_a1, matrix_a2, matrix_a3, matrix_a4 = load_associations(path, size, sp_num, stage[i], pattern)
            if i == 0:
                matrix_sps = torch.unsqueeze(matrix_sp, dim=0)
                matrix_a1s = torch.unsqueeze(matrix_a1, dim=0)
                matrix_a2s = torch.unsqueeze(matrix_a2, dim=0)
                matrix_a3s = torch.unsqueeze(matrix_a3, dim=0)
                matrix_a4s = torch.unsqueeze(matrix_a4, dim=0)
            else:
                matrix_sps = torch.cat([matrix_sps, torch.unsqueeze(matrix_sp, dim=0)], dim=0)
                matrix_a1s = torch.cat([matrix_a1s, torch.unsqueeze(matrix_a1, dim=0)], dim=0)
                matrix_a2s = torch.cat([matrix_a2s, torch.unsqueeze(matrix_a2, dim=0)], dim=0)
                matrix_a3s = torch.cat([matrix_a3s, torch.unsqueeze(matrix_a3, dim=0)], dim=0)
                matrix_a4s = torch.cat([matrix_a4s, torch.unsqueeze(matrix_a4, dim=0)], dim=0)
            # matrix_sps.append(matrix_sp)
            # matrix_a1s.append(matrix_a1)
            # matrix_a2s.append(matrix_a2)
            # matrix_a3s.append(matrix_a3)
            # matrix_a4s.append(matrix_a4)
        return matrix_sps, matrix_a1s, matrix_a2s, matrix_a3s, matrix_a4s
    else:
        matrix_sp, matrix_a1, matrix_a2, matrix_a3, matrix_a4 = load_associations(path, size, sp_num, stage, pattern)
        return matrix_sp, matrix_a1, matrix_a2, matrix_a3, matrix_a4


def load_associations(path, size, sp_num, stage, pattern):

    format = 'jpg' if path[-3:] == 'jpg' else 'jpeg'

    if stage == 99:
        path_1 = '/data2/wangzh/datasets/ISIC2018/{}_{}/T_sp/patch_a1.npy'.format(size, sp_num)
        path_2 = '/data2/wangzh/datasets/ISIC2018/{}_{}/T_sp/patch_a2.npy'.format(size, sp_num)
        path_3 = '/data2/wangzh/datasets/ISIC2018/{}_{}/T_sp/patch_a3.npy'.format(size, sp_num)
        path_4 = '/data2/wangzh/datasets/ISIC2018/{}_{}/T_sp/patch_a4.npy'.format(size, sp_num)
        path_5 = '/data2/wangzh/datasets/ISIC2018/{}_{}/T_sp/patch_a5.npy'.format(size, sp_num)
    else:
        path_1 = str.replace(path, pattern, "{}_{}/T_sp/{}/stage1".format(size, sp_num, stage))
        path_1 = str.replace(path_1, format, "npy")
        path_2 = str.replace(path, pattern, "{}_{}/T_sp/{}/stage2".format(size, sp_num, stage))
        path_2 = str.replace(path_2, format, "npy")
        path_3 = str.replace(path, pattern, "{}_{}/T_sp/{}/stage3".format(size, sp_num, stage))
        path_3 = str.replace(path_3, format, "npy")
        path_4 = str.replace(path, pattern, "{}_{}/T_sp/{}/stage4".format(size, sp_num, stage))
        path_4 = str.replace(path_4, format, "npy")
        path_5 = str.replace(path, pattern, "{}_{}/T_sp/{}/stage5".format(size, sp_num, stage))
        path_5 = str.replace(path_5, format, "npy")

    if os.path.exists(path_1):

        sp_len = int(sp_num / 4)
        association1 = np.load(path_2)
        if association1.size > sp_num:
            association1 = association1[:sp_num]
        encoded_association1 = np.zeros((association1.size, sp_len), dtype=np.int8)
        encoded_association1[np.arange(association1.size), association1] = 1
        if association1.size < sp_num:
            new_axes = np.zeros((sp_num - association1.size, sp_len), dtype=np.int8)
            encoded_association1 = np.concatenate((encoded_association1, new_axes), axis=0)  # [65536, 1024]

        encoded_sp = np.load(path_1).flatten()
        encoded_association2 = np.load(path_3)
        encoded_association3 = np.load(path_4)
        encoded_association4 = np.load(path_5)

        matrix_sp = torch.tensor(encoded_sp, dtype=torch.int)
        matrix_a1 = torch.tensor(encoded_association1, dtype=torch.float32)
        matrix_a2 = torch.tensor(encoded_association2, dtype=torch.int)
        matrix_a3 = torch.tensor(encoded_association3, dtype=torch.int)
        matrix_a4 = torch.tensor(encoded_association4, dtype=torch.int)
    else:
        print('Path {} do not exist'.format(path_1))
        return torch.zeros(1), torch.zeros(1), torch.zeros(1), torch.zeros(1), torch.zeros(1)

    return matrix_sp, matrix_a1, matrix_a2, matrix_a3, matrix_a4
