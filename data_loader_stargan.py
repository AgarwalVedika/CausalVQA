import torch
import os
import random
from torch.utils.data import Dataset
from PIL import Image
import numpy as np
from pycocotools.coco import COCO as COCOTool
import ipdb

class CocoMaskDataset(Dataset):
    def __init__(self, transform, mode):

        self.data_path = os.path.join('data','coco')
        self.transform = transform
        self.mode = mode
        filename = 'instances_train2014.json' if mode=='train2014' else  'instances_val2014.json'
        self.dataset = COCOTool(os.path.join(self.data_path, filename))
        print('Loaded Mask Data')

    def __len__(self):
        return len(self.img_ids)

    def getbyIdAndclass(self, img_id, catId, hflip=0):
        #### img_id and cls_id are integers; not list - not tensors
        maskTotal = np.zeros((self.dataset.imgs[img_id]['height'], self.dataset.imgs[img_id]['width']))
        for ann in self.dataset.loadAnns(self.dataset.getAnnIds(img_id, catId)):
            cm = self.dataset.annToMask(ann)
            maskTotal[:cm.shape[0],:cm.shape[1]] += cm
        if hflip:
            maskTotal = maskTotal[:,::-1]
        if self.transform  is not None:
            mask = torch.FloatTensor(np.asarray(self.transform(Image.fromarray(np.clip(maskTotal,0,1)))))[None,::]
        else:
            mask = torch.FloatTensor(np.asarray((Image.fromarray(np.clip(maskTotal, 0, 1)))))[None, ::]
        #plt.imshow(torchvision.utils.make_grid(mask, nrow=5).permute(1, 2, 0))
        #plt.show()
        return mask       ### returns mask of torch.Size([1, 256, 256])  ## now all being returned in original size of image- no resize by me for object removal img generation phase

    def getbyIdAndclassBatch(self, batch_size, img_ids, cls_ids, hflip=None):
        all_masks = []
        for j in range(batch_size):
            img_id = img_ids.tolist()[j][0]
            #print(img_id)
            catIds_img = cls_ids.tolist()[j]
            for i,cat_id in enumerate(catIds_img):
                if cat_id !=0 :                                    ########### this takes care of catId set to 0
                    #print(cat_id)
                    mask = self.getbyIdAndclass(img_id, cat_id, hflip)
                    all_masks.append(mask)                         ## torch.Size([1,256, 256])
        if len(all_masks) != 0:
            return(torch.stack(all_masks,dim=0))       ## torch.Size([22, 1, 256, 256])
        else:
            return None


class CocoMaskDatasetCounting(Dataset):
    def __init__(self, transform, mode):

        self.data_path = os.path.join('data','coco')
        self.transform = transform
        self.mode = mode
        filename = 'instances_train2014.json' if mode=='train2014' else  'instances_val2014.json'
        self.dataset = COCOTool(os.path.join(self.data_path, filename))
        print('Loaded Mask Data')

    def __len__(self):
        return len(self.inst_ids)

    def getbyInstanceId(self, image_id, instance_id, hflip=0):
        #### img_id and cls_id are integers; not list - not tensors

        assert self.dataset.loadAnns(instance_id)[0]['image_id'] == image_id
        mask_instance = self.dataset.annToMask(self.dataset.loadAnns(instance_id)[0])


        if hflip:
            mask_instance = mask_instance[:,::-1]
        if self.transform  is not None:
            mask = torch.FloatTensor(np.asarray(self.transform(Image.fromarray(np.clip(mask_instance,0,1)))))[None,::]
        else:
            mask = torch.FloatTensor(np.asarray((Image.fromarray(np.clip(mask_instance, 0, 1)))))[None, ::]
        #plt.imshow(torchvision.utils.make_grid(mask, nrow=5).permute(1, 2, 0))
        #plt.show()
        return mask       ### returns mask of torch.Size([1, 256, 256])  ## now all being returned in original size of image- no resize by me for object removal img generation phase

    def getbyInstanceIdBatch(self, batch_size, img_ids, inst_ids, hflip=None):
        all_masks = []
        for j in range(batch_size):
            img_id = img_ids.tolist()[j][0]
            inst_id = inst_ids.tolist()[j][0]
            #print(cat_id)
            mask = self.getbyInstanceId(img_id, inst_id, hflip)
            all_masks.append(mask)                         ## torch.Size([1,256, 256])
        if len(all_masks) != 0:
            return(torch.stack(all_masks,dim=0))       ## torch.Size([22, 1, 256, 256])
        else:
            return None
