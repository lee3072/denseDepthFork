import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
from PIL import Image
from io import BytesIO
import random
import cv2

loss_scale = 100.0
# def loadZipToMem(zip_file):
#     # Load zip file into memory
#     print('Loading dataset zip file...', end='')
#     from zipfile import ZipFile
#     input_zip = ZipFile(zip_file)
#     data = {name: input_zip.read(name) for name in input_zip.namelist()}
#     train_data_list = list((row.split(',') for row in (data['data/list_train.csv']).decode("utf-8").split('\n') if len(row) > 0))
#     eval_data_list = list((row.split(',') for row in (data['data/list_eval.csv']).decode("utf-8").split('\n') if len(row) > 0))
#     from sklearn.utils import shuffle
#     train_data_list = shuffle(train_data_list, random_state=0)
#     print('Loaded Train ({0}).'.format(len(train_data_list)))
#     eval_data_list = shuffle(eval_data_list, random_state=0)
#     print('Loaded Eval ({0}).'.format(len(eval_data_list)))
#     return data, train_data_list, eval_data_list

class depthDatasetMemory(Dataset):
    def __init__(self, data_list):
        self.datalist = data_list
        global loss_scale
        self.loss_scale = loss_scale
        # print(data_list)

    def __getitem__(self, idx):
        sample = self.datalist[idx]
        # encoded_img = np.fromstring(sample[0], dtype = np.uint8)
        # image = cv2.imdecode(encoded_img, cv2.IMREAD_COLOR)
        # print(sample)
        image = cv2.imread(sample[0])
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = np.moveaxis(image, 2, 0)
        depth = np.load(sample[1])
        # depth = cv2.resize(depth,dsize=(320,240),interpolation=cv2.INTER_AREA)
        # depth = cv2.resize(depth,dsize=(960,540),interpolation=cv2.INTER_AREA)
        depth = np.expand_dims(depth, axis=0)
        # image = torch.from_numpy(image).float()
        # depth = torch.from_numpy(depth).float()
        image = torch.from_numpy(image.astype(int)/loss_scale).float()
        depth = torch.from_numpy(depth.astype(int)/loss_scale).float()
        sample = {'image': image, 'depth': depth}
        return sample

    def __len__(self):
        return len(self.datalist)


def getTrainingTestingData(batch_size, folder):
    train_list = list()
    eval_list = list()
    with open(folder+"/list_train.csv") as f:
        for line in f:
            train_list.append(list(line.strip().split(',')))
    with open(folder+"/list_eval.csv") as f:
        for line in f:
            eval_list.append(list(line.strip().split(',')))
    # data, train_data_list, eval_data_list = loadZipToMem('realSense_data_csv.zip')
    # data, train_data_list, eval_data_list = loadZipToMem('eval_data.zip')
    transformed_training = depthDatasetMemory(train_list)
    transformed_testing = depthDatasetMemory(eval_list)

    return DataLoader(transformed_training, batch_size, shuffle=True,num_workers = 4), DataLoader(transformed_testing, batch_size, shuffle=False,num_workers = 4)
