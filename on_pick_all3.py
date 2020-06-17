import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import json
import time
import ipdb
import cv2
import os
import pickle
import nltk
import time
import pylab as plot
## set seeds
import random
random.seed(1234)
from utils_picking import my_read_old, my_read_old_area, my_read_short, intersect, round_percent, get_indices_diff_list_suffix, vqa_score_list,  worst_case_acc, ch_atleast_once
from utils_picking_thesis import my_read_old
import argparse
import matplotlib.patches as mpatches

parser = argparse.ArgumentParser()
#parser.add_argument('--model', required=True, type=str) # snmn; SAAA; CNN_LSTM
parser.add_argument('--test_split', default= 'val2014', type=str)  # val2014/edited_val2014 automatically taken care of
# parser.add_argument('-l','--list', nargs='+', help='<Required> Set flag', required=True)
# # Use like:
# # python arg.py -l 1234 2345 3456 4567

parser.add_argument('--class_label', '--list', nargs='+', default=None) # 'airplane'
parser.add_argument('--ques_type', default=None, type=list)  # 'how many'
parser.add_argument('--ans_type', default=None, type=list)  ## train2014/val2014/test2015
parser.add_argument('--dist_norm', default=1)    ## 1,2,np.inf
parser.add_argument('--model_seq', default='210', type=str)   # 210,   #TODO to use '012' you need to fix the code- image id is string or sth.. check

parser.add_argument('--snmn_lr_25_e_4', default=1, type=int)
parser.add_argument('--worst_case_ind', default=0, type=int) # 1/0


parser.add_argument('--if_flipped', default=0, type=int)


parser.add_argument('--mini_dir', default='0.1_0.0', type=str)


parser.add_argument('--if_L_norm', default=1, type=int) # 1/0
# AUTHOR'S NOTE: keep in mind: ans_vocab_list, results_val, results_edited_val- would have to be updated
# USAGE: python exp_vqa/on_pick_single.py --ans_type 'other'
#USAGE: (vqa_adv) vagarwal@d2volta10:/BS/vedika2/nobackup/thesis/code/SNMN_analysis_codes$ python FIXED_on_pick_all_3.py  --worst_case_ind 1 --class_label car bottle

##USGAE:
#(vqa_adv) vagarwal@d2pascal03:/BS/vedika2/nobackup/thesis/code/SNMN_analysis_codes$
# python FIXED_on_pick_all_3.py --mini_dir 1.0_1.0


# MODEL_0 = 'CNN_LSTM'  # MAIN_MODEL
# MODEL_1 = 'SAAA'
# MODEL_2 = 'SNMN'

def main(args):
    print()

    TEST_SPLIT = args.test_split
    #CLASS = args.class_label
    CLASSES = [ 'banana', 'person', 'sports ball', 'wine glass']
    CLASS = args.class_label
            #'toaster' , 'bottle',  'dining table', 'person',
    # chair', 'cup', 'car' , 'refrigerator', 'sandwich', 'pizza' , 'laptop', 'tennis racket', 'banana', 'orange', 'kite'
        #['sports ball', 'tennis racket' , 'frisbee', 'skis', 'kite',\
        #'bottle', 'spoon', 'knife', 'car', 'bus', 'train', 'cow', 'banana', 'person',  'pizza', 'cake', 'cup', 'bus', 'toaster', \
        #       'sandwich', 'keyboard', 'dining table', 'couch', 'bed', 'orange' ] ###args.class_label  #['car' , 'bottle']#['sandwich' , 'cake', 'pizza', 'bus'] #args.class_label  ### give  a list
    #QUES_TYPES = ['how many', 'what color is the', 'is there a', 'does the', 'does this',  'do you', 'is that a', 'how many people are in' ]#, 'how many people are', 'do you', 'can you',  'is that a', 'does this' , 'how', 'how many', 'none of the above', 'is this a', 'are there any', 'what color is the' ]
    QUES_TYPE = args.ques_type
    ANS_TYPE = args.ans_type
    DIST_NORM = args.dist_norm
    ques_types = [ 'what sport is', 'how many', 'are there any', 'what color are the', 'is there a' , 'none of the above' ,'is this person', 'is the woman', 'what is the woman', 'what is the person', 'is that a','what is the color of the']



    # if CLASS is None and QUES_TYPE is None and ANS_TYPE is None:
    #     raise ValueError('you need to specify atleast one class or ques_type or ans_type')

    # with open('setup_tool_intersection_all_3_users_indices.json') as f:
    #     all_3_users_intersection_yes = json.load(f)['flat_list_yes_intersection_indices']



    root_dir_qa = os.path.join('/BS/vedika2/nobackup/thesis/mini_datasets_qa' , str(args.mini_dir))

    coco_json = '/BS/vedika2/nobackup/snmn/coco_cat_ids.json'

    #### questions keys: 'image_id', 'question', 'question_id'
    standard_questions_val_json = '/BS/databases10/VQA_v2/Questions/v2_OpenEnded_mscoco_'  + TEST_SPLIT + '_questions.json'
    res_file_q = 'v2_OpenEnded_mscoco_' + TEST_SPLIT + '_questions.pickle'


    standard_questions_edit_val_json = os.path.join(root_dir_qa, res_file_q)


    ## ann keys: 'image_id', 'question_id', 'answers' , 'multiple_choice_answer'(the most frequent answer), 'question_type', 'answer_type'
    standard_annotations_val_json = '/BS/databases10/VQA_v2/Annotations/v2_mscoco_'  + TEST_SPLIT + '_annotations.json'
    res_file_a = 'v2_mscoco_' + TEST_SPLIT + '_annotations.json'
    standard_annotations_edit_val_json = os.path.join(root_dir_qa, res_file_a)

    img_dir= '/BS/databases10/VQA_v2/Images/' +  TEST_SPLIT + '/'
    if args.if_flipped:
        edit_img_dir = '/BS/vedika2/nobackup/thesis/flipped_edited_VQA_v2/Images/' + TEST_SPLIT + '/'
    else:
        edit_img_dir = '/BS/vedika2/nobackup/thesis/final_edited_VQA_v2/Images/' + TEST_SPLIT + '/'
    img_prefix_name = 'COCO_' + TEST_SPLIT + '_'




    with open(coco_json) as file:
        coco_f = json.load(file)
    coco_dict = {}
    for idx,details in enumerate(coco_f):
        #print(details)
        coco_dict[details['id']] = details['name']
    coco_dict_inv = {}
    for idx, details in enumerate(coco_f):
        # print(details)
        coco_dict_inv[details['name']] = details['id']

    ques_type_file = '/BS/vedika3/nobackup/VQA_helper_tools_official/QuestionTypes/mscoco_question_types.txt'
    with open(ques_type_file) as f:
        ques_type_off = f.read().splitlines()
    ans_type_off = ['yes/no', 'number', 'other']




    if args.model_seq == '012':
        ## CNN_LSTM
        model0 = 'CNN_LSTM'
        if args.if_flipped:
            results_edit_val_old = '/BS/vedika3/nobackup/pytorch-vqa/logs/flipped_edit_val_no_attn_.pickle'  # edit_val_no_attn_.pickle
        else:
            results_edit_val_old = '/BS/vedika3/nobackup/pytorch-vqa/logs/edit_val_no_attn_.pickle'

        results_val_old = '/BS/vedika3/nobackup/pytorch-vqa/logs/val_no_attn_.pickle'  # val_no_attn_.pickle
        vocab = '/BS/vedika3/nobackup/pytorch-vqa/vocab.json'
        with open(vocab, 'r') as f:
            ans_vocab = json.load(f)["answer"]
            ans_vocab_list = [k for k, v in ans_vocab.items()]
            # {v: k for k, v in ans_vocab.items()}   ### is a dictionary here but will work: keys- index- 0,1,2...

        ## Show,Ask,Attend,Answer
        model1 = 'SAAA    '
        if args.if_flipped:
            results_edit_val_old1 = '/BS/vedika3/nobackup/pytorch-vqa/logs/flipped_edit_val_with_attn_.pickle'
        else:
            results_edit_val_old1 = '/BS/vedika3/nobackup/pytorch-vqa/logs/edit_val_with_attn_.pickle'

        results_val_old1 = '/BS/vedika3/nobackup/pytorch-vqa/logs/val_with_attn_.pickle'
        vocab1 = '/BS/vedika3/nobackup/pytorch-vqa/vocab.json'
        with open(vocab1, 'r') as f:
            ans_vocab1 = json.load(f)["answer"]
            ans_vocab_list1 = [k for k, v in ans_vocab1.items()]
            # {v: k for k, v in ans_vocab.items()}   ### is a dictionary here but will work: keys- index- 0,1,2...

        ## SNMN
        model2= 'SNMN    '
        if args.if_flipped:
            results_edit_val_old2 = '/BS/vedika2/nobackup/snmn/exp_vqa/eval_outputs_vqa_v2/vqa_v2_scratch_train/vqa_v2_flipped_edited_val2014_vqa_v2_scratch_train_15000_results.pickle'
        else:
            results_edit_val_old2 = '/BS/vedika2/nobackup/snmn/exp_vqa/eval_outputs_vqa_v2/vqa_v2_scratch_train/vqa_v2_edited_val2014_vqa_v2_scratch_train_15000_results.pickle'

        results_val_old2 = '/BS/vedika2/nobackup/snmn/exp_vqa/eval_outputs_vqa_v2/vqa_v2_scratch_train/vqa_v2_val2014_vqa_v2_scratch_train_15000_results.pickle'

        if args.snmn_lr_25_e_4:
            results_edit_val_old2 = '//BS/vedika2/nobackup/snmn/exp_vqa/eval_outputs_vqa_v2/vqa_v2_scratch_train_25lr/vqa_v2_edited_val2014_vqa_v2_scratch_train_25lr_30_results.pickle'
            results_val_old2 = '/BS/vedika2/nobackup/snmn/exp_vqa/eval_outputs_vqa_v2/vqa_v2_scratch_train_25lr/vqa_v2_val2014_vqa_v2_scratch_train_25lr_30_results.pickle'

        standard_vocab_ans_file2 = '/BS/vedika2/nobackup/snmn/exp_vqa/data/answers_vqa.txt'
        with open(standard_vocab_ans_file2) as f:
            ans_vocab_list2 = f.read().splitlines()

    if args.model_seq == '210':
        # MODEL_0 = 'SNMN'  # MAIN_MODEL
        # MODEL_1 = 'SAAA'
        # MODEL_2 = 'CNN_LSTM'
        ## CNN_LSTM
        model2 = 'CNN_LSTM'
        if args.if_flipped:
            results_edit_val_old2 = '/BS/vedika3/nobackup/pytorch-vqa/logs/flipped_edit_val_no_attn_.pickle'  # edit_val_no_attn_.pickle
        else:
            results_edit_val_old2 = '/BS/vedika3/nobackup/pytorch-vqa/logs/edit_val_no_attn_.pickle'  # edit_val_no_attn_.pickle

        results_val_old2 = '/BS/vedika3/nobackup/pytorch-vqa/logs/val_no_attn_.pickle'  # val_no_attn_.pickle
        vocab2 = '/BS/vedika3/nobackup/pytorch-vqa/vocab.json'
        with open(vocab2, 'r') as f:
            ans_vocab2 = json.load(f)["answer"]
            ans_vocab_list2 = [k for k, v in ans_vocab2.items()]
            # {v: k for k, v in ans_vocab.items()}   ### is a dictionary here but will work: keys- index- 0,1,2...

        ## Show,Ask,Attend,Answer
        model1 = 'SAAA    '
        if args.if_flipped:
            results_edit_val_old1 = '/BS/vedika3/nobackup/pytorch-vqa/logs/flipped_edit_val_with_attn_.pickle'
        else:
            results_edit_val_old1 = '/BS/vedika3/nobackup/pytorch-vqa/logs/edit_val_with_attn_.pickle'

        results_val_old1 = '/BS/vedika3/nobackup/pytorch-vqa/logs/val_with_attn_.pickle'
        vocab1 = '/BS/vedika3/nobackup/pytorch-vqa/vocab.json'
        with open(vocab1, 'r') as f:
            ans_vocab1 = json.load(f)["answer"]
            ans_vocab_list1 = [k for k, v in ans_vocab1.items()]
            # {v: k for k, v in ans_vocab.items()}   ### is a dictionary here but will work: keys- index- 0,1,2...

        ## SNMN
        model0= 'SNMN    '
        if args.if_flipped:
            results_edit_val_old = '/BS/vedika2/nobackup/snmn/exp_vqa/eval_outputs_vqa_v2/vqa_v2_scratch_train/vqa_v2_flipped_edited_val2014_vqa_v2_scratch_train_15000_results.pickle'
        else:
            results_edit_val_old = '/BS/vedika2/nobackup/snmn/exp_vqa/eval_outputs_vqa_v2/vqa_v2_scratch_train/vqa_v2_edited_val2014_vqa_v2_scratch_train_15000_results.pickle'

        if args.snmn_lr_25_e_4:
            results_edit_val_old = '//BS/vedika2/nobackup/snmn/exp_vqa/eval_outputs_vqa_v2/vqa_v2_scratch_train_25lr/vqa_v2_edited_val2014_vqa_v2_scratch_train_25lr_30_results.pickle'
            results_val_old = '/BS/vedika2/nobackup/snmn/exp_vqa/eval_outputs_vqa_v2/vqa_v2_scratch_train_25lr/vqa_v2_val2014_vqa_v2_scratch_train_25lr_30_results.pickle'

        results_val_old = '/BS/vedika2/nobackup/snmn/exp_vqa/eval_outputs_vqa_v2/vqa_v2_scratch_train/vqa_v2_val2014_vqa_v2_scratch_train_15000_results.pickle'
        standard_vocab_ans_file = '/BS/vedika2/nobackup/snmn/exp_vqa/data/answers_vqa.txt'
        with open(standard_vocab_ans_file) as f:
            ans_vocab_list = f.read().splitlines()


    dir_prefix = 'all_3_models'

    qid_val, pred_ans_val, ss_vc_val, img_ids_val, ques_val, all_ans_val, ques_type_val, ans_type_val = my_read_old(
        results_val_old, standard_questions_val_json, standard_annotations_val_json)
    qid_edit_val, pred_ans_edit_val, ss_vc_edit_val, img_ids_edit_val, ques_edit_val, all_ans_edit_val, ques_type_edit_val, ans_type_edit_val ,\
         area_overlap, area_max_instance, area_total = my_read_old_area(results_edit_val_old, standard_questions_edit_val_json, standard_annotations_edit_val_json)
    len_val = len(img_ids_val)
    len_edit_val = len(img_ids_edit_val)


    qid_val1, pred_ans_val1, ss_vc_val1, img_ids_val1= my_read_short(results_val_old1, standard_questions_val_json)
    qid_edit_val1, pred_ans_edit_val1, ss_vc_edit_val1, img_ids_edit_val1 = my_read_short(results_edit_val_old1, standard_questions_edit_val_json)

    qid_val2, pred_ans_val2, ss_vc_val2, img_ids_val2= my_read_short(results_val_old2, standard_questions_val_json)
    qid_edit_val2, pred_ans_edit_val2, ss_vc_edit_val2, img_ids_edit_val2 = my_read_short(results_edit_val_old2, standard_questions_edit_val_json)

    img_ids_val = [str(i).zfill(12) for i in img_ids_val]
    img_ids_val1 = [str(i).zfill(12) for i in img_ids_val1]
    img_ids_val2 = [str(i).zfill(12) for i in img_ids_val2]

    assert(qid_val==qid_val1==qid_val2)
    assert(img_ids_val==img_ids_val1 ==img_ids_val2)
    assert(qid_edit_val==qid_edit_val1==qid_edit_val2)
    assert(img_ids_edit_val==img_ids_edit_val1==img_ids_edit_val2)


    ## creating dictionary for val set - to facilitate extensions based on q_id index
    qid_ss_predans_val = {}
    for idx, a in enumerate(qid_val):
        qid_ss_predans_val[a] = (ss_vc_val[idx], pred_ans_val[idx])
    extended_ss_vc_val = [qid_ss_predans_val[q_id][0] for q_id in qid_edit_val]
    extended_pred_ans_val = [qid_ss_predans_val[q_id][1] for q_id in qid_edit_val]
    ##  you need two ploits= |v1-v2|; max(v1[i])- v2[i][argmax[v1[i]]]
    if args.if_L_norm:
        st = time.time()
        L_norm_diff = [np.linalg.norm((extended_ss_vc_val[i] - ss_vc_edit_val[i]), ord=DIST_NORM) for i in range(len_edit_val)]
        print('time taken for calculating dist_norm_bw_ss_vc', time.time() - st)
    ## for getting the softmax score of answer predicted
    ext_ss_max_value_val = [np.max(ss_vc) for ss_vc in extended_ss_vc_val]  ## == [extended_ss_vc_val[idx][val] for idx,val in enumerate(extended_pred_ans_id_val)]
    ss_value_at_val_edit_val = [ss_vc_edit_val[idx][val] for idx,val in enumerate(extended_pred_ans_val)]  #STAYS SAME!
    diff_ss_val_label = [ext_ss_max_value_val[i] - ss_value_at_val_edit_val[i] for i in range(len_edit_val)]  ##### this is to be PLOTTED!
    #diff_ss_val_label = [np.max(extended_ss_vc_val[i]) - ss_vc_edit_val[i][np.argmax([v1[i])]]for i in range(len_edit_val)]



    qid_ss_predans_val1 = {}
    for idx, a in enumerate(qid_val1):
        qid_ss_predans_val1[a] = (ss_vc_val1[idx], pred_ans_val1[idx])
    extended_ss_vc_val1 = [qid_ss_predans_val1[q_id][0] for q_id in qid_edit_val1]
    extended_pred_ans_val1 = [qid_ss_predans_val1[q_id][1] for q_id in qid_edit_val1]
    ## you need two ploits= |v1-v2|; max(v1[i])- v2[i][argmax[v1[i]]]

    if args.if_L_norm:
        st = time.time()
        L_norm_diff1 = [np.linalg.norm((extended_ss_vc_val1[i] - ss_vc_edit_val1[i]), ord=DIST_NORM) for i in range(len_edit_val)]
        print('time taken for calculating dist_norm_bw_ss_vc', time.time() - st)
    ## for getting the softmax score of answer predicted
    ext_ss_max_value_val1 = [np.max(ss_vc) for ss_vc in extended_ss_vc_val1]  ## == [extended_ss_vc_val[idx][val] for idx,val in enumerate(extended_pred_ans_id_val)]
    ss_value_at_val_edit_val1 = [ss_vc_edit_val1[idx][val] for idx, val in enumerate(extended_pred_ans_val1)]  # STAYS SAME!
    diff_ss_val_label1 = [ext_ss_max_value_val1[i] - ss_value_at_val_edit_val1[i] for i in range(len_edit_val)]  ##### this is to be PLOTTED!

    qid_ss_predans_val2 = {}
    for idx, a in enumerate(qid_val2):
        qid_ss_predans_val2[a] = (ss_vc_val2[idx], pred_ans_val2[idx])
    extended_ss_vc_val2 = [qid_ss_predans_val2[q_id][0] for q_id in qid_edit_val2]
    extended_pred_ans_val2 = [qid_ss_predans_val2[q_id][1] for q_id in qid_edit_val2]
    ## you need two ploits= |v1-v2|; max(v1[i])- v2[i][argmax[v1[i]]]

    if args.if_L_norm:
        st = time.time()
        L_norm_diff2 = [np.linalg.norm((extended_ss_vc_val2[i] - ss_vc_edit_val2[i]), ord=DIST_NORM) for i in range(len_edit_val)]
        print('time taken for calculating dist_norm_bw_ss_vc', time.time() - st)
    ### for getting the softmax score of answer predicted
    ext_ss_max_value_val2 = [np.max(ss_vc) for ss_vc in extended_ss_vc_val2]  ## == [extended_ss_vc_val[idx][val] for idx,val in enumerate(extended_pred_ans_id_val)]
    ss_value_at_val_edit_val2 = [ss_vc_edit_val2[idx][val] for idx, val in enumerate(extended_pred_ans_val2)]  # STAYS SAME!
    diff_ss_val_label2 = [ext_ss_max_value_val2[i] - ss_value_at_val_edit_val2[i] for i in range(len_edit_val)]  ##### this is to be PLOTTED!

    ######new edit

    lab_fl_ind = [i for i in range(len_edit_val) if extended_pred_ans_val[i] != pred_ans_edit_val[i]]
    # label i.e ans was wrong before- right now- one match to 10gt ans is okay
    lab_fl_pos = [i for i in lab_fl_ind if ans_vocab_list[extended_pred_ans_val[i]] not in all_ans_edit_val[i] and ans_vocab_list[pred_ans_edit_val[i]] in all_ans_edit_val[i]]
    # label i.e ans was right before- now wrong - one match to 10gt ans is okay
    lab_fl_neg = [i for i in lab_fl_ind if ans_vocab_list[extended_pred_ans_val[i]] in all_ans_edit_val[i] and ans_vocab_list[pred_ans_edit_val[i]] not in all_ans_edit_val[i]]
    lab_fl_right = [i for i in lab_fl_ind if ans_vocab_list[extended_pred_ans_val[i]] in all_ans_edit_val[i] and ans_vocab_list[pred_ans_edit_val[i]] in all_ans_edit_val[i]]
    lab_fl_wrong = [i for i in lab_fl_ind if ans_vocab_list[extended_pred_ans_val[i]] not in all_ans_edit_val[i] and ans_vocab_list[pred_ans_edit_val[i]] not in all_ans_edit_val[i]]
    print('#labels flipped that show a positive change', len(lab_fl_pos), round_percent(len(lab_fl_pos) / len(lab_fl_ind)), '%')
    print('#labels flipped that show a negative change', len(lab_fl_neg), round_percent(len(lab_fl_neg) / len(lab_fl_ind)), '%')
    print('#labels flipped that are both right', len(lab_fl_right), round_percent(len(lab_fl_right) / len(lab_fl_ind)), '%')
    print('#labels flipped that are both wrong', len(lab_fl_wrong), round_percent(len(lab_fl_wrong) / len(lab_fl_ind)), '%')


    lab_sm_ind = [i for i in range(len_edit_val) if extended_pred_ans_val[i] == pred_ans_edit_val[i]]
    # labels that remained exactly same- so two cases possible- either right/wrong
    lab_sm_right = [i for i in lab_sm_ind if ans_vocab_list[pred_ans_edit_val[i]] in all_ans_edit_val[i]]
    lab_sm_wrong = [i for i in lab_sm_ind if ans_vocab_list[pred_ans_edit_val[i]] not in all_ans_edit_val[i]]
    print('#labels same that are both right', len(lab_sm_right), '         ', round_percent(len(lab_sm_right) / len(lab_sm_ind)), '%')
    print('#labels same that are both wrong', len(lab_sm_wrong), '         ', round_percent(len(lab_sm_wrong) / len(lab_sm_ind)), '%')

    color_vector = np.zeros(len_edit_val).tolist()
    for i in lab_fl_pos:
        color_vector[i] = 'g'         # green
    for i in lab_fl_neg:
        color_vector[i] = 'r'          # red
    for i in lab_fl_wrong:
        color_vector[i] = 'k'           # black
    for i in lab_fl_right:
        color_vector[i] = 'y'         # yellow
    for i in lab_sm_right:
        color_vector[i] = 'm'          # magenta
    for i in lab_sm_wrong:
        color_vector[i] = 'b'           # blue

    ######new edit


    for QUES_TYPE in ques_types:  # coco_dict_inv.keys():
        #ipdb.set_trace()
        #print('question type:', QUES_TYPE)

        if args.worst_case_ind:

            if len(list(set(qid_val))) != len(list(set(qid_edit_val))):
                print('not every question in orig_val made it to edit_val=> for ',
                      len(list(set(qid_val))) - len(list(set(qid_edit_val))), ' questions- no legit edited IQA possible')

            qid_gt_ans_label = {}
            for idx, a in enumerate(qid_val):
                qid_gt_ans_label[a] = all_ans_val[idx]

            qid_predans_val = {}
            for idx, a in enumerate(qid_val):
                qid_predans_val.setdefault(a, []).append(pred_ans_val[idx])

            qid_predans_idx_val = {}
            for idx, a in enumerate(qid_val):
                qid_predans_idx_val.setdefault(a, []).append(idx)

            # idx in case here refers to len(val and edit_val- order hai - so relax)
            qid_predans_edit_val = {}
            for idx, a in enumerate(qid_edit_val):
                qid_predans_edit_val.setdefault(a, []).append(pred_ans_edit_val[idx])
                # qid_predans_edit_val[a] = (pred_ans_edit_val[idx])

            qid_predans_idx_edit_val = {}
            for idx, a in enumerate(qid_edit_val):
                qid_predans_idx_edit_val.setdefault(a, []).append(idx)

                # qid_predans_imgid_edit_val = {}
            # for idx, a in enumerate(qid_edit_val):
            #     qid_predans_imgid_edit_val.setdefault(a, []).append(img_ids_edit_val[idx])

            ## creating dictionary for val set - to facilitate extensions based on q_id index

            a, b, worst_case_idx, best_case_idx, off_worst_case_idx, off_best_case_idx = \
                worst_case_acc(qid_predans_edit_val.keys(), qid_predans_idx_edit_val, qid_predans_edit_val, ans_vocab_list,
                               qid_gt_ans_label)
            # worst_case_acc(qid_predans_val.keys(), qid_predans_val,ans_vocab_list, qid_gt_ans_label)
            ch_atleast_once(qid_predans_edit_val.keys(), qid_predans_edit_val, qid_predans_val, ans_vocab_list)

            chosen_worst_indices = [worst_case_idx[ques] for ques in worst_case_idx.keys()]
            chosen_best_indices = [best_case_idx[ques] for ques in best_case_idx.keys()]
            off_chosen_worst_indices = [off_worst_case_idx[ques] for ques in off_worst_case_idx.keys()]
            off_chosen_best_indices = [off_best_case_idx[ques] for ques in off_best_case_idx.keys()]

            ########WORST INDICES######
            #chosen_indices = off_chosen_worst_indices
            #len_c = len(chosen_indices)
            #string_print = '_off_worst_indices'


            chosen_indices, chosen_diff_list_norm, chosen_diff_list_label, plt_title_suffix = get_indices_diff_list_suffix(
                coco_dict_inv, img_ids_edit_val, ques_type_edit_val, ans_type_edit_val, L_norm_diff , diff_ss_val_label, \
                CLASS =CLASS, QUES_TYPE = QUES_TYPE, ANS_TYPE = ANS_TYPE, preselected_indices = off_chosen_worst_indices, preselected_string = '_off_worst_indices')

        else:
            chosen_indices, chosen_diff_list_norm, chosen_diff_list_label, plt_title_suffix = get_indices_diff_list_suffix(
                coco_dict_inv, img_ids_edit_val, ques_type_edit_val, ans_type_edit_val, L_norm_diff, diff_ss_val_label, \
                CLASS=CLASS, QUES_TYPE=QUES_TYPE, ANS_TYPE=ANS_TYPE, preselected_indices=None, preselected_string=None)
                #preselected_indices=all_3_users_intersection_yes, preselected_string='all 3 annotators agreed')     #TODO  MAKE CHANGES HERE in case you want to look at those indices whch humans agree

        chosen_color_vector = [color_vector[i] for i in chosen_indices]

        #ipdb.set_trace()
        #chosen_indices = [i for i in chosen_indices if i in all_3_users_intersection_yes]

        ##plt_title_suffix: _airplane_how many_other_
        ### creating directories for saving interesting results

        # model_int_find_root_dir = '/BS/vedika2/nobackup/thesis/analysis_interesting_findings_models/'
        # sample_dir = model_int_find_root_dir + dir_prefix + str(plt_title_suffix) +'/'
        # os.makedirs(sample_dir, exist_ok=True)



        def onpick(event):

            index = event.ind
            print('event_ind:', index)  ## event_ind: [1394 1419] ## sensitive to the plot- so ind: 0 to len(whatever you are plotting)

            # index: 1567.8072383626895
            # event_ind: [1544 1546 1550 1551 1552 1555 1557 1558 1559 1561 1562 1563 1564 1566 ...]

            ind = chosen_indices[index[0]]                #TODO vedika so whatever is plotted: all_indices/ chosen_indices/lab_fl_neg/lab_fl_pos...and so on
            question_str = ques_edit_val[ind]

            obj_removed_id = int(img_ids_edit_val[ind])%100
            obj_removed_label = coco_dict[obj_removed_id]   ## coco_dict[int(a[0]['image_id'])%100]
            gt_answers = all_ans_edit_val[ind]
            pred_ans_before_edit = ans_vocab_list[extended_pred_ans_val[ind]]
            pred_ans_after_edit = ans_vocab_list[pred_ans_edit_val[ind]]
            conf_before = ext_ss_max_value_val[ind]
            conf_after = ss_value_at_val_edit_val[ind]
            #ipdb.set_trace()
            area_overlap_dict = area_overlap[ind]
            area_overlap_dict_label = {}
            ## use coco_dict now to make it readable
            for key in area_overlap_dict.keys():
                #print(key[0])
                ## key= (44,62)
                area_overlap_dict_label[coco_dict[key[0]], coco_dict[key[1]]]= area_overlap_dict[key]

            area_max_inst = area_max_instance[ind]
            area_tot = area_total[ind]

            pred_ans_before_edit1 = ans_vocab_list1[extended_pred_ans_val1[ind]]
            pred_ans_after_edit1 = ans_vocab_list1[pred_ans_edit_val1[ind]]
            conf_before1 = ext_ss_max_value_val1[ind]
            conf_after1 = ss_value_at_val_edit_val1[ind]

            pred_ans_before_edit2 = ans_vocab_list2[extended_pred_ans_val2[ind]]
            pred_ans_after_edit2 = ans_vocab_list2[pred_ans_edit_val2[ind]]
            conf_before2 = ext_ss_max_value_val2[ind]
            conf_after2 = ss_value_at_val_edit_val2[ind]

            print(question_str)
            print(gt_answers)
            print('pred_ans_before_edit:', pred_ans_before_edit)
            print('pred_ans_after_edit:', pred_ans_after_edit)
            print('    diff_in_label_confidence: ' , diff_ss_val_label[ind])

            #Showing images in opencv.
            org_img = img_dir + img_prefix_name + img_ids_edit_val[ind][0:12] + '.jpg'
            edit_img = edit_img_dir + img_prefix_name + img_ids_edit_val[ind] + '.jpg'
            img1 = cv2.imread(org_img)
            img2 = cv2.imread(edit_img)
            WHITE = [255,255,255]
            img1_with_border = cv2.copyMakeBorder(img1, 0, 0, 0, 20, cv2.BORDER_CONSTANT, value=WHITE)
            numpy_horizontal_concat = np.concatenate((img1_with_border,img2), axis=1)
            height, width, _ = numpy_horizontal_concat.shape
            img1 = numpy_horizontal_concat
            vcat = cv2.copyMakeBorder(img1, 180, 10, 50,50, cv2.BORDER_CONSTANT, value=WHITE)   ## 100,100
            #cv2.imshow('constant', vcat)
            #white = np.zeros((250, 600, 3), np.uint8)  ## img1.shape[1]
            #white[:] = (255, 255, 255)
            #vcat = cv2.vconcat((white, img1))
            # if img_ids_edit_val[ind] == '000000147921_000000000016' and qid_edit_val[ind] == '147921006':
            #     ipdb.set_trace()

            str1 = str(question_str) + '    object_removed: ' + str(obj_removed_label) + '   image_id:' + str(img_ids_edit_val[ind]) + 'ques_id: ' + str(qid_edit_val[ind])

            #ipdb.set_trace()
            str2 = 'model' + '       ans_bef' + '    ans_aft' + '  conf aft_ans_before' +  '  diff_label_conf'  + ' l1_norm_diff'

            str3 = model0 +  '   {}:'.format(pred_ans_before_edit) + str(round(conf_before,3)) + '   ' + \
                      '   {}:'.format(pred_ans_after_edit) + str(round(np.max(ss_vc_edit_val[ind]),3)) + \
                      '   {}:'.format(pred_ans_before_edit) + str(round(conf_after,3))  + \
                      '    ' + str(round(diff_ss_val_label[ind], 3)) + \
                      '    ' + str(round(L_norm_diff[ind],3))

            str4 = model1  + '   {}:'.format(pred_ans_before_edit1) + str(round(conf_before1,3)) + '   ' + \
                      '   {}:'.format(pred_ans_after_edit1) + str(round(np.max(ss_vc_edit_val1[ind]),3)) + \
                      '   {}:'.format(pred_ans_before_edit1) + str(round(conf_after1,3))  + \
                      '    ' + str(round(diff_ss_val_label1[ind], 3)) + \
                      '    ' + str(round(L_norm_diff1[ind],3))

            str5 = model2  + '   {}:'.format(pred_ans_before_edit2) + str(round(conf_before2,3)) + '   ' + \
                      '   {}:'.format(pred_ans_after_edit2) + str(round(np.max(ss_vc_edit_val2[ind]),3)) + \
                      '   {}:'.format(pred_ans_before_edit2) + str(round(conf_after2,3))  + \
                      '    ' + str(round(diff_ss_val_label2[ind], 3)) + \
                      '    ' + str(round(L_norm_diff2[ind],3))

            str6 = 'all_gt_answers: ' + str(gt_answers)

            # str_put = str(area_overlap_dict_label)
            # len_str_put_third = int(len(str_put)/3)
            #if len_str_put<1000:
            # str8 =  'area_overlap_dict: ' + str_put[0:len_str_put_third]
            # str9 =  '                   ' + str_put[len_str_put_third: 2*len_str_put_third]
            # str10 = '                   ' + str_put[2*len_str_put_third:]

            print('area_max_instance({}) : '.format(str(obj_removed_label)) + str(area_max_inst) + '    area_all_instances_included({}): '.format(str(obj_removed_label)) + str(area_tot))
            print('area_overlap_dict: ',str(area_overlap_dict_label) )

            font_sz = 0.50
            cv2.putText(vcat, str1, (30,25), cv2.FONT_HERSHEY_SIMPLEX, font_sz, (0,0,0), 1, cv2.LINE_AA)
            cv2.putText(vcat, str2, (30,50), cv2.FONT_HERSHEY_SIMPLEX, font_sz, (0, 0, 0), 1, cv2.LINE_AA)
            cv2.putText(vcat, str3, (30,75), cv2.FONT_HERSHEY_SIMPLEX, font_sz, (0, 0, 0), 1, cv2.LINE_AA)
            cv2.putText(vcat, str4, (30,100), cv2.FONT_HERSHEY_SIMPLEX, font_sz, (0, 0, 0), 1, cv2.LINE_AA)
            cv2.putText(vcat, str5, (30,125), cv2.FONT_HERSHEY_SIMPLEX, font_sz, (0, 0, 0), 1, cv2.LINE_AA)
            cv2.putText(vcat, str6, (30, 150), cv2.FONT_HERSHEY_SIMPLEX, font_sz, (0, 0, 0), 1, cv2.LINE_AA)
            # cv2.putText(vcat, str7, (30, 180), cv2.FONT_HERSHEY_SIMPLEX, font_sz, (0, 0, 0), 1, cv2.LINE_AA)
            # cv2.putText(vcat, str8, (30, 210), cv2.FONT_HERSHEY_SIMPLEX, font_sz, (0, 0, 0), 1, cv2.LINE_AA)
            # cv2.putText(vcat, str9, (30, 240), cv2.FONT_HERSHEY_SIMPLEX, font_sz, (0, 0, 0), 1, cv2.LINE_AA)
            # cv2.putText(vcat, str10, (30, 270), cv2.FONT_HERSHEY_SIMPLEX, font_sz, (0, 0, 0), 1, cv2.LINE_AA)
            #cv2.putText(numpy_horizontal_concat, str1, (10, 600), cv2.FONT_HERSHEY_SIMPLEX, 0.1, (255, 255, 255), 2, cv2.LINE_AA)

            cv2.imshow('img', vcat)

            k= cv2.waitKey(0)
            if k==27: # wait for ESC key to exit
                cv2.destroyWindow('img')
            return

        ## figure with on_pick
        #print(plt.rcParams.get('figure.figsize'))
        # default_figsize_x = plt.rcParams.get('figure.figsize')[0]
        # default_figsize_y = plt.rcParams.get('figure.figsize')[1]
        # fig = plt.figure(figsize=(2 * default_figsize_x, 2 * default_figsize_y));
        fig = plt.figure(figsize=(12,8))
        ax = plt.subplot();
        # ax.set_title('click to see; |ss_vc_val - ss_vc_edit_val|')
        ax.set_title('click to see, CNN_LSTM_diff_in_softmax_value: {}'.format(plt_title_suffix))
        # L_norm_diff, diff_ss_val_label
        fig.canvas.mpl_connect('pick_event', onpick)
        ax.scatter(chosen_indices, chosen_diff_list_label, alpha=0.8, c=chosen_color_vector, s=30, picker=True)
        # ax.legend(bbox_to_anchor=(1.0, 1.0))  # (1.1, 1.05))
        patch1 = mpatches.Patch(color='g', label='lab_fl_pos')
        patch2 = mpatches.Patch(color='r', label='lab_fl_neg')
        patch3 = mpatches.Patch(color='k', label='lab_fl_wrong')
        patch4 = mpatches.Patch(color='y', label='lab_fl_right')
        patch5 = mpatches.Patch(color='m', label='lab_sm_right')
        patch6 = mpatches.Patch(color='b', label='lab_sm_wrong')
        plt.legend(handles=[patch1, patch2, patch3, patch4, patch5, patch6], bbox_to_anchor=(1.0, 1.0))
        plt.xlabel('indices')
        # plt.ylabel('diff_softmax_val_to_edit_val')
        plt.show()
        # fig.savefig('{}/diff_softmax_value_CNN_LSTM_{}.png'.format(sample_dir, plt_title_suffix))
        # print('saving plot to dir', sample_dir)

if __name__ == '__main__':
    args = parser.parse_args()
    main(args)
