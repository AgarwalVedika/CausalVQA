import json
import time
import config
from pycocotools.coco import COCO
import numpy as np
import os
import pickle
from skimage.morphology import disk, square
import scipy


selem = disk(2)
square_dil = square(5)
# >>> disk(2)
# array([[0, 0, 1, 0, 0],
#        [0, 1, 1, 1, 0],
#        [1, 1, 1, 1, 1],
#        [0, 1, 1, 1, 0],
#        [0, 0, 1, 0, 0]], dtype=uint8)

# >>> square(5)
# array([[1, 1, 1, 1, 1],
#        [1, 1, 1, 1, 1],
#        [1, 1, 1, 1, 1],
#        [1, 1, 1, 1, 1],
#        [1, 1, 1, 1, 1]], dtype=uint8)


question_file_1 = config.vqa_q_dir +  "v2_OpenEnded_mscoco_train2014_questions.json"
question_file_2 = config.vqa_q_dir + "Questions/v2_OpenEnded_mscoco_val2014_questions.json"
question_file_3 = config.vqa_q_dir + "v2_OpenEnded_mscoco_test-dev2015_questions.json"
question_file_4 = config.vqa_q_dir + "v2_OpenEnded_mscoco_test2015_questions.json"

## just getting the list of words- index doesnt matter here
with open("coco-labels/coco_labels_80_classes.txt") as f:
    # return the split results, which is all the words in the file.
    coco_80 = f.read().splitlines()  ###coco 80 classes

# with open('all_coco.txt') as f:
#     # return the split results, which is all the words in the file.
#     coco_stuff_91= f.read().splitlines()      ##coco_stuff 91 classes


ann_coco_file_val = config.coco_ann_dir + 'instances_val2014.json'
ann_coco_file_train = config.coco_ann_dir + 'instances_train2014.json'
with open(ann_coco_file_train) as f:
    ann_coco_1 = json.load(f)['categories']

id_list = []
class_list = []
for cate in ann_coco_1:
    id_list.append(cate['id'])
    class_list.append(cate['name'])
    

def overlap_criterion(obj_q, cls_img_set, maskTotal_per_cat_id_all_ins_dil):
    # OVERLAP CRITERION ## how much of other object interferes with the nouns_q/nouns_ans object
    ### if object referred in question overlaps with any object in the image
    ## dont discard that other object
    masks_intersect = {}
    cls_img_set_dummy = cls_img_set[:]
    if len(obj_q)!=0:
        for cls_id in obj_q:
            mask_q = maskTotal_per_cat_id_all_ins_dil[cls_id]
            #cls_img_set_dummy.remove(cls_id)
            if len(cls_img_set_dummy)!=0:
                for cat_id in cls_img_set_dummy:
                    if cls_id != cat_id:  ## as {(1, 1): 1.0; class overlaps with itself completely
                        mask_other = maskTotal_per_cat_id_all_ins_dil[cat_id]  ## mask_other= object whihc could be removed
                        score = round(np.sum(mask_q * mask_other) / (np.sum(mask_q)),3)  # round(np.sum(mask_q * mask_other) / (np.sum(mask_q) + np.sum(mask_other)),3)
                        if score:
                            masks_intersect[cls_id, cat_id] = score
        return masks_intersect
    else:
        return None

    # ipdb > intersect_dict_ind_sq5  : don't want (1,1):1 class overlaps with itself completely
    # {(1, 1): 1.0, (1, 8): 0.046, (1, 15): 0.002, (1, 27): 0.011, (1, 31): 0.018, (1, 41): 0.027, (1, 67): 0.001,
    #  (5, 5): 1.0, (8, 1): 0.384, (8, 8): 1.0, (8, 27): 0.039, (15, 1): 0.003, (15, 15): 1.0, (15, 67): 0.21,
    #  (27, 1): 0.809, (27, 8): 0.351, (27, 27): 1.0, (31, 1): 1.0, (31, 31): 1.0, (41, 1): 0.486, (41, 41): 1.0,
    #  (67, 1): 0.004, (67, 15): 0.981, (67, 67): 1.0}

    ### creating directories for saving interesting results


def overlap_criterion_mask_flip_mask(obj_q, cls_img_set,flipped_maskTotal_per_cat_id_all_ins_dil, maskTotal_per_cat_id_all_ins_dil):  ### object_to be removed is flipped now- other object is original
    # OVERLAP CRITERION ## how much of other object interferes with the nouns_q/nouns_ans object
    ### if object referred in question overlaps with any object in the image
    ## dont discard that other object
    masks_intersect = {}
    cls_img_set_dummy = cls_img_set[:]
    if len(obj_q)!=0:
        for cls_id in obj_q:
            mask_q = maskTotal_per_cat_id_all_ins_dil[cls_id]
            #cls_img_set_dummy.remove(cls_id)
            if len(cls_img_set_dummy)!=0:
                for cat_id in cls_img_set_dummy:
                    #if cls_id != cat_id:  ## as {(1, 1): 1.0; class overlaps with itself completely => no longer the case with one set flipped
                    mask_other = flipped_maskTotal_per_cat_id_all_ins_dil[cat_id]   ## mask_other= object whihc could be removed
                    score = round(np.sum(mask_q * mask_other) / (np.sum(mask_q)),3)  # round(np.sum(mask_q * mask_other) / (np.sum(mask_q) + np.sum(mask_other)),3)
                    if score:
                        masks_intersect[cls_id, cat_id] = score
        #ipdb.set_trace()
        return masks_intersect
    else:
        return None

    # ipdb > masks_intersect
    # {(1, 1): 0.7, (1, 8): 0.054, (1, 27): 0.013, (1, 31): 0.012, (1, 41): 0.033, (8, 1): 0.455, (8, 41): 0.144,
    #  (15, 15): 0.808, (15, 67): 0.16, (27, 1): 1.0, (31, 1): 0.706, (41, 1): 0.592, (41, 8): 0.311, (41, 41): 0.136,
    #  (67, 15): 0.748, (67, 67): 0.559}


def area_criterion(coco, img_id, classes_img_set):
    # AREA CRITERION
    ### discarding criteria- if any in the percent_area_per_catId_all_inst_included- is more than 0.5-
    ## dont discard that object!!
    ### EVER DECIDE TO DO PER INSTANCE-  [coco.loadAnns(i) for i in coco.getAnnIds(img_id)][i][0]['area']
    ### getting the are occupied: area_masks_per_cls_all_instances_included
    #ipdb.set_trace()

    area_img = coco.imgs[img_id]['height'] * coco.imgs[img_id]['width']
    percent_area_masks_per_cls_all_instances_included = {}
    percent_area_masks_per_cls_max_area_instance = {}
    # maskTotal_per_cat_id_all_ins = {}
    maskTotal_per_cat_id_all_ins_dilated_square5 = {}
    maskTotal_per_cat_id_all_ins_dilated_default = {}
    flipped_maskTotal_per_cat_id_all_ins_dilated_square5 = {}
    flipped_maskTotal_per_cat_id_all_ins_dilated_default = {}
    iou_mask_flipped_mask = {}
    for cat_id in classes_img_set:  ## finding masks for person
        maskTotal = np.zeros((coco.imgs[img_id]['height'], coco.imgs[img_id]['width']))  ## initializing here
        diff_areas = []
        for ann in coco.loadAnns(coco.getAnnIds(img_id, cat_id)):
        # >> > for ann in coco.loadAnns(coco.getAnnIds(img_id, cat_id)):
        #     ...
        #     print(ann)
        # ...
        # {'segmentation': [
        #     [243.46, 385.7, 262.61, 362.91, 257.14, 328.26, 244.37, 298.17, 235.25, 275.37, 228.87, 264.43, 249.84,
        #      244.37, 281.75, 243.46, 310.02, 264.43, 332.82, 284.49, 367.47, 289.05, 393.0, 289.96, 407.59, 318.23,
        #      415.79, 338.29, 424.0, 355.61, 462.3, 373.85, 486.0, 378.41, 501.51, 361.08, 492.39, 335.55, 482.36,
        #      311.85, 468.68, 294.52, 457.74, 273.55, 463.21, 262.61, 485.09, 252.58, 500.59, 234.34, 512.45, 194.22,
        #      513.36, 172.34, 510.62, 152.28, 558.04, 143.16, 593.6, 119.45, 569.89, 103.04, 548.92, 80.24, 506.06,
        #      69.3, 477.8, 57.45, 445.88, 42.86, 418.53, 33.74, 381.14, 24.62, 351.97, 17.32, 329.17, 12.77, 319.14,
        #      8.21, 280.84, 0.91, 183.28, 0.91, 54.71, 4.56, 2.74, 1.82, 10.94, 139.51, 36.47, 165.95, 79.33, 193.31,
        #      129.48, 235.25, 155.92, 281.75, 168.69, 331.91, 175.98, 366.55, 183.28, 382.97]],
        #  'area': 137505.95750000002, 'iscrowd': 0, 'image_id': 25138, 'bbox': [2.74, 0.91, 590.86, 384.79],
        #  'category_id': 17, 'id': 48366}
            cm = coco.annToMask(ann)
            maskTotal[:cm.shape[0], :cm.shape[1]] += cm  ### adding all person masks to one single mask
            diff_areas.append(ann['area'])


        flipped_maskTotal= np.flip(maskTotal,axis=1)  #(maskTotal_flipped)
        iou_mask_flipped_mask[cat_id] = round(np.sum(maskTotal*flipped_maskTotal) / (np.sum(maskTotal) + np.sum(flipped_maskTotal)),3)
        # plt.imshow(maskTotal)
        # plt.show()
        # plt.imshow(flipped_maskTotal)
        # plt.show()

        max_area_instance_per_catId  = np.max(diff_areas)
        percent_area_masks_per_cls_all_instances_included[cat_id] = round((np.sum(maskTotal) / area_img), 3)
        percent_area_masks_per_cls_max_area_instance[cat_id]= round((max_area_instance_per_catId / area_img), 3)
        # maskTotal_per_cat_id_all_ins[cat_id] = maskTotal
        maskTotal_per_cat_id_all_ins_dilated_square5[cat_id] = scipy.ndimage.morphology.binary_dilation(maskTotal, square_dil) ## use same diltion sturcure as in GAN
        maskTotal_per_cat_id_all_ins_dilated_default[cat_id] = scipy.ndimage.morphology.binary_dilation(maskTotal)  ## use default diltion sturcure in scipy.ndimage.binary_dilation
        #maskTotal_per_cat_id_all_ins_dilated_5[cat_id] = scipy.ndimage.morphology.binary_dilation(maskTotal, selem)

        flipped_maskTotal_per_cat_id_all_ins_dilated_square5[cat_id] = scipy.ndimage.morphology.binary_dilation(flipped_maskTotal, square_dil) ## use same diltion sturcure as in GAN
        flipped_maskTotal_per_cat_id_all_ins_dilated_default[cat_id] = scipy.ndimage.morphology.binary_dilation(flipped_maskTotal)  ## use default diltion sturcure in scipy.ndimage.binary_dilation

    return percent_area_masks_per_cls_all_instances_included, percent_area_masks_per_cls_max_area_instance,  \
           maskTotal_per_cat_id_all_ins_dilated_square5, maskTotal_per_cat_id_all_ins_dilated_default, \
           flipped_maskTotal_per_cat_id_all_ins_dilated_square5, flipped_maskTotal_per_cat_id_all_ins_dilated_default, iou_mask_flipped_mask



def ready_json(question_file, res_pkl_file, coco):
    start = time.time();

    abcd = []
    file_data = {}

    with open(question_file) as f:
        questions = json.load(f)['questions']  ##read json
    for n_q, q in enumerate(questions):
        print(n_q)
        classes_img = [coco.loadAnns(i)[0]['category_id'] for idx, i in enumerate(coco.getAnnIds(q['image_id']))]
        classes_img_set = sorted(list(set(classes_img)))
        percent_area_masks_per_cls_all_instances, percent_area_masks_per_cls_max_area_instance, \
        maskTotal_per_cat_id_all_ins_dil_sq5, maskTotal_per_cat_id_all_ins_dil_default, \
        flipped_maskTotal_per_cat_id_all_ins_dil_sq5, flipped_maskTotal_per_cat_id_all_ins_dil_default, iou_mask_flipped_mask = area_criterion(coco, q['image_id'], classes_img_set)

        intersect_dict_ind_sq5 = overlap_criterion(classes_img_set, classes_img_set, maskTotal_per_cat_id_all_ins_dil_sq5)
        intersect_dict_ind_default = overlap_criterion(classes_img_set, classes_img_set, maskTotal_per_cat_id_all_ins_dil_default)

        flipped_intersect_dict_ind_sq5 = overlap_criterion_mask_flip_mask(classes_img_set, classes_img_set, flipped_maskTotal_per_cat_id_all_ins_dil_sq5, maskTotal_per_cat_id_all_ins_dil_sq5)
        flipped_intersect_dict_ind_default = overlap_criterion_mask_flip_mask(classes_img_set, classes_img_set, flipped_maskTotal_per_cat_id_all_ins_dil_default, maskTotal_per_cat_id_all_ins_dil_default)

        abcd.append({"image_id": q['image_id'],
                     "classes_img" : classes_img,
                     "percent_area_per_catId_all_inst": percent_area_masks_per_cls_all_instances, # Area of TotalMask/area of image #(round((np.sum(maskTotal) / area_img), 3))
                     "percent_area_per_catId_max_inst":percent_area_masks_per_cls_max_area_instance , # Area of max_obj/area of image #(round((np.sum(maskTotal) / area_img), 3))
                     "if_intersect_overlap_sq5": intersect_dict_ind_sq5,
                     "if_intersect_overlap_default": intersect_dict_ind_default,
                     "iou_mask_flipped_mask":iou_mask_flipped_mask,
                     "flipped_if_intersect_overlap_sq5": flipped_intersect_dict_ind_sq5,
                     "flipped_if_intersect_overlap_default": flipped_intersect_dict_ind_default})  # (A1*A2/A1) A1 and A2 are areas of dilated object masks A1 and A2
                                                                    #  #round(np.sum(mask_q * mask_other) / (np.sum(mask_q))),3)
        #ipdb.set_trace()
    file_data['area_and_intersection'] = abcd
    res_pkl = os.path.join(res_pkl_file)
    os.makedirs(res_pkl, exist_ok=True)
    # sample_dir = model_int_find_root_dir  + dir_prefix + str(plt_title_suffix) +'/'
    # os.makedirs(sample_dir, exist_ok=True)
    with open(res_pkl, 'wb') as f:
        pickle.dump(file_data,f, pickle.HIGHEST_PROTOCOL)
    print('saving pkl complete')
    print(time.time() - start)


coco_train = COCO(ann_coco_file_train)
ready_json(question_file_1,'./coco_areas_and_intersection/coco_vqa_train2014.json', coco_train)

coco_val = COCO(ann_coco_file_val)
# coco_val.getCatIds('sports ball')
ready_json(question_file_2, './coco_areas_and_intersection/coco_vqa_val2014.json',  coco_val)


# ready_json(question_file_3, 'tagged_test_dev_questions.json', nouns_list_3)
# ready_json(question_file_4, 'tagged_test_questions.json', nouns_list_4)
