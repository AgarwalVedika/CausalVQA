import json
import os
from PIL import Image

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.dataloader import default_collate
import torchvision.transforms as transforms
from my_snippets import pad


# ### following clevr-iep code style : https://github.com/facebookresearch/clevr-iep/blob/master/iep/data.py
# ## simpler to follow- https://github.com/yunjey/pytorch-tutorial/blob/master/tutorials/03-advanced/image_captioning/data_loader.py


class imgDataset(Dataset):
    """VQA dataset"""

    def __init__(self, new_image_ids, new_classes_img_all, image_dir, mode, transform=None):

        self.image_dir = image_dir
        self.transform = transform
        self.new_images_ids = new_image_ids
        self.new_classes_img_all = new_classes_img_all
        self.mode = mode

        v = []
        for a in self.new_classes_img_all:
            v.append(len(a))
        self.max_v = max(v)

    def __len__(self):
        return len(self.new_images_ids)

    def __getitem__(self, idx):
        """Returns ONE data pair-image,match_coco_objects"""

        # print('Reading image data')
        img_id = self.new_images_ids[idx]
        #print(img_id)
        image = Image.open(os.path.join(self.image_dir, 'COCO_' + self.mode + '_' + str(img_id).zfill(12) + '.jpg')).convert('RGB')
        if self.transform is not None:
            image = self.transform(image)
        classes_img = self.new_classes_img_all[idx]

        pad(classes_img, self.max_v, padding=0)

        # print('converting data to Tensor')

        img_id = torch.LongTensor([img_id])
        classes_img = torch.LongTensor(classes_img)

        return image, classes_img, img_id


def imgDataLoader(**kwargs):
    if 'new_image_ids' not in kwargs:
        raise ValueError('Must give list(set(image_ids))')
    if 'new_classes_img_all' not in kwargs:
        raise ValueError('Must give set(cls) in each')

    new_image_ids = kwargs.pop('new_image_ids')
    new_classes_img_all = kwargs.pop('new_classes_img_all')
    image_dir = kwargs.pop('image_dir')
    mode = kwargs.pop('mode')
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5),
                             (0.5, 0.5, 0.5))])
    dataset = imgDataset(new_image_ids, new_classes_img_all, image_dir, mode, transform)
    kwargs['collate_fn'] = default_collate
    data_loader = DataLoader(dataset, **kwargs)

    return data_loader



class imgDataset_Counting(Dataset):
    """VQA dataset"""

    def __init__(self, new_image_ids, target_instance_ids, image_dir, mode, transform=None):

        self.image_dir = image_dir
        self.transform = transform
        self.new_images_ids = new_image_ids
        self.target_instance_ids = target_instance_ids
        self.mode = mode

    def __len__(self):
        return len(self.new_images_ids)

    def __getitem__(self, idx):
        """Returns ONE data pair-image,match_coco_objects"""

        # print('Reading image data')
        img_id = self.new_images_ids[idx]
        #print(img_id)
        image = Image.open(os.path.join(self.image_dir, 'COCO_' + self.mode + '_' + str(img_id).zfill(12) + '.jpg')).convert('RGB')
        if self.transform is not None:
            image = self.transform(image)
        target_instance_ids = self.target_instance_ids[idx]

        # print('converting data to Tensor')

        img_id = torch.LongTensor([img_id])
        classes_img = torch.LongTensor(target_instance_ids)

        return image, target_instance_ids, img_id

def imgDataLoaderCounting(**kwargs):
    if 'new_image_ids' not in kwargs:
        raise ValueError('Must give list(set(image_ids))')
    if 'target_instance_ids' not in kwargs:
        raise ValueError('Must give target_instance_ids')

    new_image_ids = kwargs.pop('new_image_ids')
    target_instance_ids = kwargs.pop('target_instance_ids')
    image_dir = kwargs.pop('image_dir')
    mode = kwargs.pop('mode')
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5),
                             (0.5, 0.5, 0.5))])
    dataset = imgDataset(new_image_ids, target_instance_ids, image_dir, mode, transform)
    kwargs['collate_fn'] = default_collate
    data_loader = DataLoader(dataset, **kwargs)

    return data_loader


#sanity checks:
# loader = VQADataLoader(**loader_kwargs)
# len(loader) = size of validation set= #questions / batch size

# Visualize an example -  loader works- perfectly well- able to load batch size- no issues
# out = loader.dataset[3]
# out[0].shape
# ## to visualize
# from my_models import show
# show(out[0])
# image_dir = '/BS/databases10/VQA_v2/Images/val2014/'
# Image.open(os.path.join(image_dir, 'COCO_val2014_' + str(out[4].tolist()[0]).zfill(12) + '.jpg'))