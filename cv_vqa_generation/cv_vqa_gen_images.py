from __future__ import absolute_import, division, print_function
import os
import argparse
import time
import pickle
from pycocotools.coco import COCO
import ipdb
import config
import json
from my_snippets import show2, repeated_images
from my_snippets import visualizing_images_masks_batch
from object_remover import ObjectRemover
from data_loader_custom import imgDataLoaderCounting
from data_loader_stargan import CocoMaskDatasetCounting
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

    root_output_dir = config.cv_images_dir
    root_image_dir = config.vqa_images_dir
    output_dir = os.path.join(root_output_dir, input_mode + '/')
    image_dir = os.path.join(root_image_dir, input_mode + '/')

    ann_coco_file = config.coco_ann_dir + 'instances_' + input_mode + '.json'
    coco = COCO(ann_coco_file)

    if output_dir is not None:
        if not os.path.exists(output_dir):
            os.makedirs(os.path.dirname(output_dir))

    with open(input_mode+ 'coco_counting_id_area_overlap_only_one_considered_at_a_time.pickle', 'rb') as file:
        id_area_overlap = pickle.load(file)
    instance_ids_target = []
    all_imgs_counting = []
    target_count = 0
    for mini_details in id_area_overlap:
        all_imgs_counting.append(mini_details['image_id'])
        overlap_score = mini_details['overlap_score_with_rest']
        if isinstance(overlap_score, float) or overlap_score == 'only_candidate_instance':
            if isinstance(overlap_score, float) and overlap_score <= 0.0:
                # print(mini_details)
                target_count += 1
                instance_ids_target.append(mini_details['id'])
            else:
                target_count += 1
                instance_ids_target.append(mini_details['id'])
    ##### TEH BELOW NUMBERS ARE NOT NECESSARILY RIGHT!!!
    # print('so new edited #IQA:', target_count)
    # print('unique images in counting edited set:', len(set(instance_ids_target)))
    # print('#counting IQA in original set:', len(id_area_overlap))
    # print('unique images in counting original set:', len(set(all_imgs_counting)))


    target_instance_ids = list(set(instance_ids_target))
    target_instance_ids_list = [[i] for i in target_instance_ids]
    corresponding_target_image_ids = [coco.loadAnns(inst_id)[0]['image_id'] for inst_id in target_instance_ids]
    corresponding_target_cat_ids = [coco.loadAnns(inst_id)[0]['category_id'] for inst_id in target_instance_ids]

    loader_kwargs = {
        'new_image_ids': corresponding_target_image_ids,
        'target_instance_ids': target_instance_ids_list,
        'image_dir': image_dir,
        'mode': input_mode ,
        'batch_size': batch_size,
        'shuffle': False,
    }

    loader = imgDataLoaderCounting(**loader_kwargs)
    gtMaskDataset = CocoMaskDatasetCounting(transform=None, mode=input_mode)
    removal_pretrained = config.removal_model_512
    remover = ObjectRemover(removal_model=removal_pretrained, dilateMask=5)

    if args.use_gpu == 1:
        remover = remover.cuda()

    for i_batch, batch in enumerate(loader):
        print(i_batch/len(loader))
        images, target_instance_ids, img_ids = batch  #dataset = imgDataset(new_image_ids, target_instance_ids, image_dir, mode, transform)

        if args.use_gpu == 1:
            images = images.cuda()
            target_instance_ids = target_instance_ids.cuda()
            img_ids = img_ids.cuda()
        b_size = img_ids.size()[0]   ### to take care of the last batch- as it won't be of batch_size, so use the adaptive batch size
        gtMasks = gtMaskDataset.getbyInstanceIdBatch(b_size, img_ids,target_instance_ids, hflip=0)  # torch.Size([22, 1, 256, 256])
        rep_images = repeated_images(b_size, images, target_instance_ids)
        if gtMasks is not None:
            if args.use_gpu == 1:
                gtMasks = gtMasks.cuda()
            edited_images = remover(rep_images, gtMasks)
            #visualizing_images_masks_batch(rep_images, edited_images, gtMasks)
            id_list = repeated_image_list(b_size,img_ids,target_instance_ids)
            target_list = final_target_list(b_size, target_instance_ids)
            save_image_batch(id_list,target_list, edited_images, output_dir, input_mode)
    print('total time taken:', time.time()-start, 'for all the images in', input_mode, 'batch size used was 1')

if __name__ == '__main__':
    args = parser.parse_args()
    main(args)
