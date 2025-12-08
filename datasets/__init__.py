import os
import glob
import time
from importlib import import_module
from multiprocessing.pool import Pool
from multiprocessing import Process
from pathlib import Path
from skimage.measure import label
import cv2
import numpy as np
import torch
import torch.nn.functional as F
from skimage.segmentation import mark_boundaries, slic
from sklearn.preprocessing import normalize
from sklearn.cluster import KMeans
from tqdm import tqdm
import scipy

from datasets.superpixel_generator import format_sub_sp, save_graph

max_length = 0


def load_dataset(args, fold=0, train=True):
    print("=> creating dataset [{}], fold {}...".format(args.dataset_name, fold))
    m = import_module("datasets." + args.dataset_name.lower())
    if train:
        # training mode
        train_dataset, val_dataset = m.load_dataset(args, fold, train)
        return train_dataset, val_dataset
    else:
        # testing mode
        test_dataset = m.load_dataset(args, fold, train)
        return test_dataset


def largestConnectComponent(bw_img, ):
    '''
    compute largest Connect component of a binary image

    Parameters:
    ---

    bw_img: ndarray
        binary image

    Example:
    ---
        >>> lcc = largestConnectComponent(bw_img)

    '''

    labeled_img, num = label(bw_img, background=0, return_num=True)

    max_label = 0
    max_num = 0
    for i in range(1, num + 1):
        if np.sum(labeled_img == i) > max_num:
            max_num = np.sum(labeled_img == i)
            max_label = i
    lcc = (labeled_img == max_label)

    lcc = scipy.ndimage.binary_fill_holes(lcc).astype(int)

    return lcc


def reformat_label(sp):
    sp = np.array(sp, np.int64)
    H, W = sp.shape
    ls = list(set(np.reshape(sp, [-1]).tolist()))

    dic = {}
    for i in range(len(ls)):
        dic[ls[i]] = i

    new_sp = sp
    for i in range(H):
        for j in range(W):
            new_sp[i, j] = dic[new_sp[i, j]]
    return new_sp


def find_max_length(corenal_list):
    global max_length
    sp_num = 256
    size = 224

    raw = cv2.imread(corenal_list)
    raw = cv2.resize(raw, (size, size))
    sp_mask = slic(raw, n_segments=sp_num, compactness=10, start_label=1)

    dim = np.amax(sp_mask)

    kernel = np.ones((3, 3), np.uint8)

    length = 0
    for i in range(1, dim + 1):
        mask_temp = np.zeros_like(sp_mask, dtype=np.uint8)
        mask_temp[np.where(sp_mask == i)] = 1
        pos_expand = cv2.dilate(src=mask_temp, kernel=kernel, iterations=1)

        sp_temp = sp_mask[np.where(pos_expand == 1)]
        sp_unique = np.unique(sp_temp)

        length += len(sp_unique)

    if length > max_length:
        max_length = length
    print(max_length)


def format_graph(sp_mask, sp_num):

    # dim = np.amax(sp_mask)
    # if format == 'dim':
    #     graph = np.zeros((2, 1899), dtype=np.int64)
    # else:
    #     graph = np.zeros((dim, dim), dtype=np.uint8)
    # kernel = np.ones((3, 3), np.uint8)
    # d = 0
    # for i in range(1, dim + 1):
    #     mask_temp = np.zeros_like(sp_mask, dtype=np.uint8)
    #     mask_temp[np.where(sp_mask == i)] = 1
    #     pos_expand = cv2.dilate(src=mask_temp, kernel=kernel, iterations=1)

    #     sp_temp = sp_mask[np.where(pos_expand == 1)]
    #     sp_unique = np.unique(sp_temp)

    #     if format == 'dim':
    #         for k in range(len(sp_unique)):
    #             graph[0][d] = i
    #             graph[1][d] = sp_unique[k]
    #             d += 1
    #     else:
    #         graph[i - 1][sp_unique - 1] = 1

    # return graph

    A = np.zeros([sp_num, sp_num], dtype=np.float32)
    (h, w) = sp_mask.shape
    for i in range(h - 2):
        for j in range(w - 2):
            sub = sp_mask[i:i + 2, j:j + 2]
            sub_max = np.max(sub).astype(np.int32)
            sub_min = np.min(sub).astype(np.int32)
            if sub_max != sub_min:
                idx1 = sub_max
                idx2 = sub_min
                if A[idx1, idx2] != 0:
                    continue

                # pix1 = self.S[idx1]
                # pix2 = self.S[idx2]
                # diss = np.exp(-np.sum(np.square(pix1 - pix2)) / sigma ** 2)
                A[idx1, idx2] = A[idx2, idx1] = 1

    edges = np.zeros((2, 8500), dtype=np.int64)
    edge_indices = np.nonzero(A)
    len_edges = len(edge_indices[0])
    edges[:, :len_edges] = edge_indices

    return edges


def format_graph_label(target, sp, sp_num):

    cls_label = np.zeros(sp_num, dtype=np.int64)
    for i in range(sp_num):
        sp_index = np.where(sp == i)
        if np.sum(target[sp_index]) > 0:
            cls_label[i] = 1
        else:
            cls_label[i] = 0

    return cls_label


def format_association(raw, sp, size, sp_num):
    raw = np.reshape(raw, (-1, 3))
    sp = np.reshape(sp, (-1, 1))

    Q = np.zeros([size**2, sp_num], dtype=np.bool_)

    for i in range(sp_num):
        idx = np.where(sp == i)[0]
        Q[idx, i] = True

    # norm_Q = normalize(Q, norm='l1', axis=0)
    return Q


def norm(x):
    x = np.array(x)
    return (x - np.amin(x) + 1) / (np.amax(x) - np.amin(x) + 1) * 255


def calculate_similiarity(raw, img_size, sp_num, index, index2, index3, index4, index5):
    img_level = False
    if img_level:
        similiarity = np.zeros((img_size ** 2), dtype=np.float32)
        similiarity2 = np.zeros((img_size ** 2), dtype=np.float32)
        similiarity3 = np.zeros((img_size ** 2), dtype=np.float32)
        similiarity4 = np.zeros((img_size ** 2), dtype=np.float32)
        similiarity5 = np.zeros((img_size ** 2), dtype=np.float32)
    else:
        similiarity = np.zeros((img_size ** 2), dtype=np.float32)
        similiarity2 = np.zeros((sp_num), dtype=np.float32)
        similiarity3 = np.zeros((sp_num // 4), dtype=np.float32)
        similiarity4 = np.zeros((sp_num // 16), dtype=np.float32)
        similiarity5 = np.zeros((sp_num // 64), dtype=np.float32)

    association = np.zeros((index.size, sp_num), dtype=np.int8)
    association[np.arange(index.size), index] = 1

    association2 = np.zeros((index2.size, sp_num // 4), dtype=np.int8)
    association2[np.arange(index2.size), index2] = 1
    if index2.size < sp_num:
        new_axes = np.zeros((sp_num - index2.size, sp_num // 4), dtype=np.int8)
        association2 = np.concatenate((association2, new_axes), axis=0)

    association3 = np.zeros((index3.size, sp_num // 16), dtype=np.int8)
    association3[np.arange(index3.size), index3] = 1

    association4 = np.zeros((index4.size, sp_num // 64), dtype=np.int8)
    association4[np.arange(index4.size), index4] = 1

    association5 = np.zeros((index5.size, sp_num // 256), dtype=np.int8)
    association5[np.arange(index5.size), index5] = 1

    raw_t = torch.from_numpy(raw).float().flatten(0, 1).permute(1, 0)

    if img_level:
        association_t = torch.from_numpy(association).float()
        association2_t = torch.matmul(association_t, torch.from_numpy(association2).float())
        association3_t = torch.matmul(association2_t, torch.from_numpy(association3).float())
        association4_t = torch.matmul(association3_t, torch.from_numpy(association4).float())
        association5_t = torch.matmul(association4_t, torch.from_numpy(association5).float())
    else:
        association_t = torch.from_numpy(association).float()
        association2_t = torch.from_numpy(association2).float()
        association3_t = torch.from_numpy(association3).float()
        association4_t = torch.from_numpy(association4).float()
        association5_t = torch.from_numpy(association5).float()

    association_t = F.normalize(association_t, p=1.0, dim=0)
    association2_t = F.normalize(association2_t, p=1.0, dim=0)
    association3_t = F.normalize(association3_t, p=1.0, dim=0)
    association4_t = F.normalize(association4_t, p=1.0, dim=0)
    association5_t = F.normalize(association5_t, p=1.0, dim=0)

    # [65536, 3] * [65536, 1024] -> [1024, 3]
    if img_level:
        merged_img = torch.matmul(raw_t, association_t)
        merged_img2 = torch.matmul(raw_t, association2_t)
        merged_img3 = torch.matmul(raw_t, association3_t)
        merged_img4 = torch.matmul(raw_t, association4_t)
        merged_img5 = torch.matmul(raw_t, association5_t)
    else:
        merged_img = torch.matmul(raw_t, association_t)
        merged_img2 = torch.matmul(merged_img, association2_t)
        merged_img3 = torch.matmul(merged_img2, association3_t)
        merged_img4 = torch.matmul(merged_img3, association4_t)
        merged_img5 = torch.matmul(merged_img4, association5_t)

    raw_t = F.normalize(raw_t.permute(1, 0), dim=1)
    merged_img = F.normalize(merged_img.permute(1, 0), dim=1)
    merged_img2 = F.normalize(merged_img2.permute(1, 0), dim=1)
    merged_img3 = F.normalize(merged_img3.permute(1, 0), dim=1)
    merged_img4 = F.normalize(merged_img4.permute(1, 0), dim=1)
    merged_img5 = F.normalize(merged_img5.permute(1, 0), dim=1)

    association2_t = torch.matmul(association_t, torch.from_numpy(association2).float())
    association3_t = torch.matmul(association2_t, torch.from_numpy(association3).float())
    association4_t = torch.matmul(association3_t, torch.from_numpy(association4).float())
    association5_t = torch.matmul(association4_t, torch.from_numpy(association5).float())
    association_t = torch.argmax(association_t, dim=1).numpy()
    association2_t = torch.argmax(association2_t, dim=1).numpy()
    association3_t = torch.argmax(association3_t, dim=1).numpy()
    association4_t = torch.argmax(association4_t, dim=1).numpy()
    association5_t = torch.argmax(association5_t, dim=1).numpy()

    if img_level:
        for i in range(img_size ** 2):
            similiarity[i] = (raw_t[i] @ merged_img[association_t[i]].T).numpy()

        for i in range(img_size ** 2):
            similiarity2[i] = (raw_t[i] @ merged_img2[association2_t[i]].T).numpy()

        for i in range(img_size ** 2):
            similiarity3[i] = (raw_t[i] @ merged_img3[association3_t[i]].T).numpy()

        for i in range(img_size ** 2):
            similiarity4[i] = (raw_t[i] @ merged_img4[association4_t[i]].T).numpy()

        for i in range(img_size ** 2):
            similiarity5[i] = (raw_t[i] @ merged_img5[association5_t[i]].T).numpy()

        similiarity_view = np.reshape(similiarity, (img_size, img_size))
        similiarity_view2 = np.reshape(similiarity2, (img_size, img_size))
        similiarity_view3 = np.reshape(similiarity3, (img_size, img_size))
        similiarity_view4 = np.reshape(similiarity4, (img_size, img_size))
        similiarity_view5 = np.reshape(similiarity5, (img_size, img_size))

    else:
        for i in range(img_size ** 2):
            similiarity[i] = (raw_t[i] @ merged_img[index[i]].T).numpy()

        for i in range(index2.size):
            similiarity2[i] = (merged_img[i] @ merged_img2[index2[i]].T).numpy()

        for i in range(sp_num // 4):
            similiarity3[i] = (merged_img2[i] @ merged_img3[index3[i]].T).numpy()

        for i in range(sp_num // 16):
            similiarity4[i] = (merged_img3[i] @ merged_img4[index4[i]].T).numpy()

        for i in range(sp_num // 64):
            similiarity5[i] = (merged_img4[i] @ merged_img5[index5[i]].T).numpy()

        similiarity_view = np.reshape(similiarity, (img_size, img_size))
        similiarity_view2 = np.zeros((img_size ** 2), dtype=np.float32)
        similiarity_view3 = np.zeros((img_size ** 2), dtype=np.float32)
        similiarity_view4 = np.zeros((img_size ** 2), dtype=np.float32)
        similiarity_view5 = np.zeros((img_size ** 2), dtype=np.float32)

        for i in range(sp_num):
            similiarity_view2[np.where(association_t == i)] = similiarity2[i]
        for i in range(sp_num // 4):
            similiarity_view3[np.where(association2_t == i)] = similiarity3[i]
        for i in range(sp_num // 16):
            similiarity_view4[np.where(association3_t == i)] = similiarity4[i]
        for i in range(sp_num // 64):
            similiarity_view5[np.where(association4_t == i)] = similiarity5[i]

        similiarity_view2 = np.reshape(similiarity_view2, (img_size, img_size))
        similiarity_view3 = np.reshape(similiarity_view3, (img_size, img_size))
        similiarity_view4 = np.reshape(similiarity_view4, (img_size, img_size))
        similiarity_view5 = np.reshape(similiarity_view5, (img_size, img_size))

    sim = [similiarity, similiarity2, similiarity3, similiarity4, similiarity5]
    sim_view = [similiarity_view, similiarity_view2, similiarity_view3, similiarity_view4, similiarity_view5]

    return sim, sim_view


