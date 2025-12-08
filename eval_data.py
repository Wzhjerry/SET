import numpy as np
import cv2
import glob
from tqdm import tqdm

# data = np.load('/data2/wangzh/datasets/ISIC2018/256_256/T_sp/6/stage1/ISIC_0010863.npy')


for seed in [30, 27, 25, 23, 21, 20, 18, 16, 15, 13, 11, 10, 8, 6, 5]:
    print(seed)
    data_path1 = glob.glob("/data2/wangzh/datasets/ISIC2018/256_256/T_sp/{}/stage1/*.npy".format(seed))
    data_path2 = glob.glob("/data2/wangzh/datasets/ISIC2018/256_256/T_sp/{}/stage2/*.npy".format(seed))
    data_path3 = glob.glob("/data2/wangzh/datasets/ISIC2018/256_256/T_sp/{}/stage3/*.npy".format(seed))
    data_path4 = glob.glob("/data2/wangzh/datasets/ISIC2018/256_256/T_sp/{}/stage4/*.npy".format(seed))
    data_path5 = glob.glob("/data2/wangzh/datasets/ISIC2018/256_256/T_sp/{}/stage5/*.npy".format(seed))

    for path in tqdm(data_path1):
        data = np.load(path)

        if np.max(data) > 255:
            print(np.max(data))
            print(path)
            data[np.where(data > 255)] = 255
            np.save(path, data)
