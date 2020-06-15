import torch
import torchvision
import numpy as np
import matplotlib.pyplot as plt
import scipy.misc as sc
import os

def pad(seq, target_length, padding=0):
    """Extend the sequence seq with padding (default: 0) so as to make
    its length up to target_length. Return seq. If seq is already
    longer than target_length, raise TooLongError.
    """
    length = len(seq)
    if length > target_length:
        raise ValueError("sequence too long ({}) for target length {}"
                           .format(length, target_length))
    seq.extend([padding] * (target_length - length))
    return seq


def show(img):
    if img.type() == 'torch.cuda.FloatTensor':
        img = img.cpu()
    npimg = img.numpy()
    plt.figure(figsize=(10,10))
    plt.imshow(((np.transpose(npimg, (1,2,0))+1.0)*255./2.0).astype(np.uint8), interpolation='nearest')
    plt.show()

def show2(img):
    if img.type() == 'torch.cuda.FloatTensor':
        img = img.cpu()
    npimg = img.numpy()
    return (((np.transpose(npimg, (1,2,0))+1.0)*255./2.0).astype(np.uint8))    ### default it's nearest- use this to plot normal images

# plt.figure(figsize=(10,10))
# plt.imshow(torchvision.utils.make_grid(gtMask, nrow=5).permute(1, 2, 0))


def total_count(batch_size,target_cls_ids):
    count_batch = 0
    for i in range(batch_size):
        for j in target_cls_ids.tolist()[i]:
            if j!=0:
                count_batch+=1
    return (count_batch)


def repeated_images(batch_size, imgs, target_cls_ids):
    all_images = []
    for i in range(batch_size):
        #count_img = 0
        for j in target_cls_ids.tolist()[i]:
            if j!= 0:
                all_images.append(imgs[i])
    if len(all_images) != 0:
        return torch.stack(all_images,dim=0)
    else:
        return None


def repeated_image_list(batch_size, img_ids, target_cls_ids):
    img_list = []
    for i in range(batch_size):
        #count_img = 0
        for j in target_cls_ids.tolist()[i]:
            if j!= 0:
                img_list.append(img_ids.tolist()[i][0])
    return(img_list)



def repeated_images_qids(batch_size, imgs, q_ids, target_cls_ids):
    all_images = []
    all_ques = []
    for i in range(batch_size):
        #count_img = 0
        for j in target_cls_ids.tolist()[i]:
            if j!= 0:
                all_images.append(imgs[i])
                all_ques.append(q_ids[i])
    if len(all_ques) != 0:
        return (torch.stack(all_images,dim=0), torch.stack(all_ques,dim=0))
    else:
        return (None,None)
#a = images[0]
#a.repeat(6,1,1,1)
#rep_images = repeated_images(batch_size,images,targets_to_be_removed)


def final_target_list(batch_size,target_cls_ids):
    target_list= []
    for i in range(batch_size):
        for j in target_cls_ids[i]:
            if j!=0:
                target_list.append(j.tolist())
    return target_list


def save_image_batch(id_list,target_list,edited_images,output_dir, mode):
    for i in range(len(id_list)):
        new_i_id = str(id_list[i]).zfill(12) + '_' + str(target_list[i]).zfill(12)
        filename = os.path.join(output_dir, 'COCO_' + mode +'_' + new_i_id + '.jpg')
        sc.imsave(filename, show2(edited_images[i]))


def visualizing_images_masks_batch(true_rep_imgs, edited_imgs, masks):
    ## all these three are tensors of [22= repeat_count, 3/1, 256,256]
    plt.axis('off')
    for i in range(masks.size()[0]):
        fig = plt.figure(figsize=(15, 15))
        a = fig.add_subplot(1, 3, 1)
        imgplot = plt.imshow(show2(true_rep_imgs[i]))
        a.set_title('Before')
        a = fig.add_subplot(1, 3, 2)
        imgplot = plt.imshow(show2(edited_imgs[i]))
        a.set_title('After')
        a = fig.add_subplot(1, 3, 3)
        imgplot = plt.imshow(torchvision.utils.make_grid(masks[i], nrow=5).permute(1, 2, 0))
        a.set_title('Mask')


# def edited_images_plots(img_id, class_id_to_be_removed):
#     target_to_be_removed_2 = torch.LongTensor([class_id_to_be_removed])
#
#     gtMask2 = gtMaskDataset.getbyIdAndclass(img_id, target_to_be_removed_2)
#     a2 = torch.Tensor(batch_size, 1, image_size[0], image_size[1])
#     a2[:] = gtMask2
#     edited_image_2 = remover(images, a2)
#     fig = plt.figure(figsize=(15, 15))
#     a = fig.add_subplot(1, 3, 1)
#     imgplot = plt.imshow(show2(images[0]))
#     a.set_title('Before')
#     a = fig.add_subplot(1, 3, 2)
#     imgplot = plt.imshow(show2(edited_image_2[0]))
#     a.set_title('After')
#     a = fig.add_subplot(1, 3, 3)
#     imgplot = plt.imshow(torchvision.utils.make_grid(gtMask2, nrow=5).permute(1, 2, 0))
#     # imgplot = plt.imshow(show2(edited_image_2[0]))
#
#     a.set_title('Mask')

#     show(images[0])
#     show(edited_image_2[0])
#     plt.figure(figsize=(10,10))
#     plt.imshow(torchvision.utils.make_grid(gtMask2, nrow=5).permute(1, 2, 0))