from torch.utils.data import Dataset
import cv2
import numpy as np
import os
import torch
from skimage.transform import resize
class MyTraindataset(Dataset):
    def __init__(self, train_path,img_size=416):
        with open(train_path, 'r') as file:
            self.img_files = file.readlines()
        self.label_files = [path.replace('images', 'labels').replace('.png', '.txt').replace('.jpg', '.txt') for path in self.img_files]
        self.img_shape = (img_size, img_size)
        self.max_objects = 213

    def __getitem__(self, item):
        # get image
        img_path = self.img_files[item % len(self.img_files)].rstrip()
        img = cv2.imread(img_path)

        ori_h, ori_w= img.shape[0], img.shape[1]
        dim_diff = np.abs(ori_w-ori_h)
        # Upper (left) and lower (right) padding
        pad1, pad2 = dim_diff // 2, dim_diff - dim_diff // 2
        # Determine padding
        pad = ((pad1, pad2), (0, 0), (0, 0)) if ori_h <= ori_w else ((0, 0), (pad1, pad2), (0, 0))
        # Add padding
        # Get new shape after padding
        input_img = np.pad(img, pad, 'constant', constant_values=128) / 255.
        padded_h, padded_w, _ = input_img.shape
        # Resize and normalize
        input_img = resize(input_img, (self.img_shape[0],self.img_shape[1], 3), mode='reflect')
        # Channels-first
        input_img = np.transpose(input_img, (2, 0, 1))
        # As pytorch tensor
        input_img = torch.from_numpy(input_img).float()

        # get label
        label_path = self.label_files[item % len(self.img_files)].rstrip()
        labels = None
        if os.path.exists(label_path):
            labels = np.loadtxt(label_path).reshape(-1, 5)
            x1 = ori_w * (labels[:, 1] - labels[:, 3] / 2)
            y1 = ori_h * (labels[:, 2] - labels[:, 4] / 2)
            x2 = ori_w * (labels[:, 1] + labels[:, 3] / 2)
            y2 = ori_h * (labels[:, 2] + labels[:, 4] / 2)
            # Adjust for added padding
            x1 += pad[1][0]
            y1 += pad[0][0]
            x2 += pad[1][0]
            y2 += pad[0][0]
            labels[:, 1] = ((x1 + x2) / 2) / padded_w
            labels[:, 2] = ((y1 + y2) / 2) / padded_h
            labels[:, 3] *= ori_w/padded_w
            labels[:, 4] *= ori_h/padded_h
        # Fill matrix
        filled_labels = np.zeros((self.max_objects, 5))
        if labels is not None:
            filled_labels[range(len(labels))[:self.max_objects]] = labels[:self.max_objects]
        filled_labels = torch.from_numpy(filled_labels)
        return img_path, input_img, filled_labels

    def __len__(self):
        return  len(self.img_files)