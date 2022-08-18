import os
from torch.utils.data import Dataset
import numpy as np
import torch


Model_Flag = "2s-CNN"  # 2s-CNN, 3DJP-CNN, 3DRJDP-CNN
motion_label = {"0": 0, "1": 1, "2": 2, "3": 3}

num_frame = 109
num_joint = 20

maxnum = 0
minnum = 0

def alignment(path_motion):
    temp = np.loadtxt(path_motion, delimiter=",")
    temp = temp[:, 2:77]  # 0: frame index, 1: time

    # remove end effectors such as toes_end and align motion by linear scaling
    temp = torch.from_numpy(temp)
    temp = np.delete(temp, [15, 16, 17, 27, 28, 29, 42, 43, 44, 57, 58, 59, 72, 73, 74], axis=1)
    temp = temp.view(1, len(temp), 60)
    temp = temp.permute(0, 2, 1)
    temp = torch.nn.functional.interpolate(temp, size=num_frame, mode='linear')
    temp = temp.permute(0, 2, 1)
    temp = temp.view(num_frame, 60)

    temp = temp.numpy()
    ma = np.max(temp)
    mi = np.min(temp)
    global maxnum
    global minnum
    if maxnum < ma:
        maxnum = ma
    if minnum > mi:
        minnum = mi
    return temp

class MotionDataset(Dataset):
    def __init__(self, data_dir, trans_form=None):
        self.label_name = motion_label
        self.motion_info = self.get_motion_info(data_dir)  # motion_info stores paths and labels of motions
        self.transform = trans_form

    def get_motion_feature_with_alignment(self, motion):

        temp = motion
        temp = np.array(temp)

        # normalization
        temp = (temp - minnum)/(maxnum-minnum)
        temp = (temp) / (maxnum)

        # 3DRJDP feature (20*19, 109, 3)
        xx = list()
        rjdp = list()
        for i in range(0, 60, 3):
            x1 = temp[:, i:i+3]
            for j in range(0, 60, 3):
                if i == j:
                    continue
                x2 = temp[:, j:j+3]
                x = x1 - x2
                rjdp.append(x)

        # joint position feature (20, 109, 3)
        for i in range(0, 60, 3):
            x = temp[:, i:i + 3]
            xx.append(x)

        res = torch.Tensor(xx)
        res = res.type(torch.FloatTensor)
        res2 = torch.Tensor(rjdp)
        res2 = res2.type(torch.FloatTensor)

        return res, res2

    def __getitem__(self, index):
        motion, label = self.motion_info[index]
        motion, motion2 = self.get_motion_feature_with_alignment(motion)
        if Model_Flag == "2s-CNN":
            return motion, motion2, label
        elif Model_Flag == "3DJP-CNN":
            return motion, label
        elif Model_Flag == "3DRJDP-CNN":
            return motion2, label

    def __len__(self):
        return len(self.motion_info)

    @staticmethod
    def get_motion_info(data_dir):
        data_info = list()
        motion_info = list()
        motion_names = os.listdir(data_dir[0])
        for mo in range (len(motion_names)):
            motion_info.append((motion_names[mo], data_dir[0]))

        for i in range(1, len(data_dir)):
            motion_names= os.listdir(data_dir[i])
            motion_names = list(filter(lambda x: x.endswith('.trc'), motion_names))
            for mo in range(len(motion_names)):
                motion_info.append((motion_names[mo], data_dir[i]))

        # get path for each motion
        for i in range(len(motion_info)):
            motion_name = motion_info[i][0]
            path_motion = os.path.join(motion_info[i][1], motion_name)
            motion = alignment(path_motion)
            label = int(motion_name[0])
            data_info.append((motion, int(label)))
        return data_info

