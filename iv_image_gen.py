from __future__ import absolute_import, division, print_function
import os
import argparse
import time

from object_remover import ObjectRemover

import torch
import torchvision.transforms as transforms
from data_loader_stargan import CocoMaskDataset
from pycocotools.coco import COCO
import scipy.misc as sc
import json
from PIL import Image
from my_snippets import show2, repeated_images
from my_snippets import visualizing_images_masks_batch
from data_loader_custom import imgDataLoader
from data_loader_stargan import CocoMaskDataset
from my_snippets import final_target_list, repeated_image_list, save_image_batch

parser = argparse.ArgumentParser()
parser.add_argument('--use_gpu', default=1, type=int)
parser.add_argument('--input_mode', required= True, type=str)  ## train2014/val2014/test2015

######### batch_size =1!!- not conditioned on image_size1!
def main(args):
    print()

    start = time.time()

    batch_size = 1   ### fixed it is
    input_mode = args.input_mode
    print(input_mode)

    root_output_dir = '/BS/vedika2/nobackup/thesis/flipped_edited_VQA_v2/'
    root_image_dir = '/BS/databases10/VQA_v2/Images/'
    output_dir = os.path.join(root_output_dir, input_mode + '/')
    image_dir = os.path.join(root_image_dir, input_mode + '/')

    images_json_path = 'coco_classes_' + input_mode + '_images.json'
    image_ids = json.load(open(images_json_path, 'r'))['image_ids']### images_ids are redundant
    classes_ids_img = json.load(open(images_json_path, 'r'))['classes_ids_img']### images_ids are redundant

    ann_coco_file_val = '/BS/databases/coco/annotations/instances_' + input_mode + '.json'
    coco = COCO(ann_coco_file_val)
    if output_dir is not None:
        if not os.path.exists(output_dir):
            os.makedirs(os.path.dirname(output_dir))

    new_image_ids = list(set(image_ids))
    classes_img_all = []
    for image_id in new_image_ids:
        ann_id_list = coco.getAnnIds(image_id)
        classes_img = [] # classes_img = [coco.loadAnns(i)[0]['category_id'] for idx, i in enumerate(coco.getAnnIds(img_id))]
        for each_class_id in ann_id_list:
            #print(each_class_id)
            for details in coco.loadAnns(each_class_id):
                classes_img.append(details['category_id'])
        classes_img_all.append(classes_img)
    print(len(new_image_ids))
    print(len(classes_img_all))


    new_classes_img_all= []
    for i,cls in enumerate(classes_img_all):
        cls_set_list = list(set(cls))           ## you are doing set operations here- in case you want to find instance wise- dont do set
        new_classes_img_all.append(cls_set_list)
    print(len(new_classes_img_all))

    loader_kwargs = {
        'new_image_ids': new_image_ids,
        'new_classes_img_all': new_classes_img_all ,
        'image_dir': image_dir,
        'mode': input_mode ,
        'batch_size': batch_size,
        'shuffle': False,
    }

    loader = imgDataLoader(**loader_kwargs)
    gtMaskDataset = CocoMaskDataset(transform=None, mode=input_mode)

    # # Pre-trained models
    # 128
    # x128
    # resoluion --> / BS / rshetty - wrk / work / code / controlled - generation / trained_models_2 / stargan / fulleditor / final_models /
    # checkpoint_stargan_coco_fulleditor_LowResMask_pascal_RandDiscrWdecay_wgan_30pcUnion_noGT_imnet_V2_msz32_ftuneMask_withPmask_L1150_tv_nb4_styleloss3k_248_1570.pth.tar
    #
    # 256
    # x256
    # resolution --> / BS / rshetty - wrk / work / code / controlled - generation / trained_models_2 / stargan / fulleditor /
    # checkpoint_stargan_coco_fulleditor_wgan_50pcUnion_msz32_withPmask_L1150_style3k_tv_nb4_512sz_sqznet_mgpu_b14_255_3483.pth.tar
    #
    # 512
    # x512
    # resolution --> / BS / rshetty - wrk / work / code / controlled - generation / trained_models_2 / stargan / fulleditor /
    # checkpoint_stargan_coco_fulleditor_wgan_40pcUnion_msz32_withPmask_L1150_tv_nb4_256sz_sqznet_250_1951.pth.tar

    removal_pretrained = '/BS/rshetty-wrk/work/code/controlled-generation/trained_models_2/stargan/fulleditor/' \
                         'checkpoint_stargan_coco_fulleditor_wgan_50pcUnion_msz32_withPmask_L1150_style3k_tv_nb4_512sz_sqznet_mgpu_b14_255_3483.pth.tar'
    remover = ObjectRemover(removal_model=removal_pretrained, dilateMask=5)

    if args.use_gpu == 1:
        remover = remover.cuda()

    for i_batch, batch in enumerate(loader):
        print(i_batch/len(loader))
        images, classes_imgs, img_ids = batch
        if args.use_gpu == 1:
            images = images.cuda()
            classes_imgs = classes_imgs.cuda()
            img_ids = img_ids.cuda()
        b_size = img_ids.size()[0]   ### to take care of the last batch- as it won't be of batch_size, so use the adaptive batch size
        gtMasks = gtMaskDataset.getbyIdAndclassBatch(b_size, img_ids,classes_imgs, hflip=0)  # torch.Size([22, 1, 256, 256])
        rep_images = repeated_images(b_size, images, classes_imgs)
        if gtMasks is not None:
            if args.use_gpu == 1:
                gtMasks = gtMasks.cuda()
            edited_images = remover(rep_images, gtMasks)
            #visualizing_images_masks_batch(rep_images, edited_images, gtMasks)
            id_list = repeated_image_list(b_size,img_ids,classes_imgs)
            target_list = final_target_list(b_size, classes_imgs)
            save_image_batch(id_list,target_list, edited_images, output_dir, input_mode)
    print('total time taken:', time.time()-start, 'for all the images in', input_mode, 'batch size used was 1')

if __name__ == '__main__':
    args = parser.parse_args()
    main(args)
