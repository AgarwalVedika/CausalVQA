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
from utils_picking import intersect, round_percent, get_indices_diff_list_suffix, vqa_score_list,  worst_case_acc, ch_atleast_once
from utils_picking_thesis import my_read_old
import argparse
import matplotlib.patches as mpatches

parser = argparse.ArgumentParser()
#parser.add_argument('--model', required=True, type=str) # snmn; SAAA; CNN_LSTM
parser.add_argument('--only_baseline_visualization', default=0, type=int)  # in case of 0.1_0.0

parser.add_argument('--label_check', default='lab_fl_pos', type=str)  #   'lab_sm_right', 'lab_sm_wrong', 'lab_fl_pos', 'lab_fl_neg' , 'lab_fl_wrong', 'lab_fl_ind', 'lab_sm_ind',

parser.add_argument('--answer_not_zero_one', default=0, type=int)  # in case of 0.1_0.0  

parser.add_argument('--check_spcific_indices_by_img_id', default=0, type=int)  # in case of 0.1_0.0  check_spcific_indices_by_img_id= it overwrites everything
parser.add_argument('--comparison_against_baseline', default=0, type=int)  # nothing to do with only_baseline_visualization - rather in baseline/ft_orig/ft_orig_edit: you want ft_orig_edit comparsion against baseline/ft_orig
parser.add_argument('--snmn_baseline_lr_25e4', default=0, type=int)
parser.add_argument('--test_split', default= 'val2014', type=str)  # val2014/edited_val2014 automatically taken care of
parser.add_argument('--dist_norm', default=1)    ## 1,2,np.inf
parser.add_argument('--model_seq', default='210', type=str)   # 210,   #TODO to use '012' you need to fix the code- imaage id is string or sth.. check
parser.add_argument('--if_L_norm', default=1, type=int) # 1/0




# MODEL_0 = 'CNN_LSTM'  # MAIN_MODEL
# MODEL_1 = 'SAAA'
# MODEL_2 = 'SNMN'

def main(args):
    print()

    TEST_SPLIT = 'val2014'
    ques_type = 'counting'
    mode = TEST_SPLIT
    DIST_NORM = args.dist_norm
    comparison_against_baseline = args.comparison_against_baseline

    orig_root_dir_qa = '/BS/vedika2/nobackup/thesis/mini_datasets_qa_CNN_finetune_testing/' + ques_type + '/orig_90_10/'
    edit_root_dir_qa = '/BS/vedika2/nobackup/thesis/mini_datasets_qa_CNN_finetune_testing/' + ques_type + '/del_1/'
    coco_json = '/BS/vedika2/nobackup/snmn/coco_cat_ids.json'

    #### questions keys: 'image_id', 'question', 'question_id'
    res_file_q = 'v2_OpenEnded_mscoco_' + TEST_SPLIT + '_questions.json'
    standard_questions_val_json = os.path.join(orig_root_dir_qa, res_file_q)
    standard_questions_edit_val_json = os.path.join(edit_root_dir_qa, res_file_q)

    ## ann keys: 'image_id', 'question_id', 'answers' , 'multiple_choice_answer'(the most frequent answer), 'question_type', 'answer_type'
    res_file_a = 'v2_mscoco_' + TEST_SPLIT + '_annotations.json'
    standard_annotations_val_json = os.path.join(orig_root_dir_qa, res_file_a)
    standard_annotations_edit_val_json = os.path.join(edit_root_dir_qa, res_file_a)


    img_dir= '/BS/vedika3/nobackup/pytorch-vqa/mscoco/' +  TEST_SPLIT + '/'

    edit_img_dir = '/BS/vedika2/nobackup/thesis/IMAGES_counting_del1_edited_VQA_v2/'+ mode + '/'
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



    if args.snmn_baseline_lr_25e4:
        indices_dir_snmn_lr = '/BS/vedika2/nobackup/thesis/code/SNMN_analysis_codes/indices_snmn_25e4_lr'
    else:
        indices_dir_snmn_lr = '/BS/vedika2/nobackup/thesis/code/SNMN_analysis_codes/indices_snmn_old_lr_1e_3'

    indices_file = os.path.join( indices_dir_snmn_lr , 'indices_all_flips_models_counting_del1.pickle')
    with open(indices_file, 'rb') as f:
        indices_labels = pickle.load(f)


    ## CNN_LSTM
    model0 = 'CNN_LSTM'

    results_edit_val_old = indices_labels['CNN_LSTM']['Baseline']['results_edit_val_old']
    results_val_old = indices_labels['CNN_LSTM']['Baseline']['results_val_old']
    vocab = '/BS/vedika3/nobackup/pytorch-vqa/vocab.json'
    with open(vocab, 'r') as f:
        ans_vocab = json.load(f)["answer"]
        ans_vocab_list = [k for k, v in ans_vocab.items()]
        # {v: k for k, v in ans_vocab.items()}   ### is a dictionary here but will work: keys- index- 0,1,2...


    vocab1 = '/BS/vedika3/nobackup/pytorch-vqa/vocab.json'
    with open(vocab1, 'r') as f:
        ans_vocab1 = json.load(f)["answer"]
        ans_vocab_list1 = [k for k, v in ans_vocab1.items()]
        # {v: k for k, v in ans_vocab.items()}   ### is a dictionary here but will work: keys- index- 0,1,2...


    standard_vocab_ans_file2 = '/BS/vedika2/nobackup/snmn/exp_vqa/data/answers_vqa.txt'
    with open(standard_vocab_ans_file2) as f:
            ans_vocab_list2 = f.read().splitlines()



    qid_val, pred_ans_val, ss_vc_val, img_ids_val, ques_val, all_ans_val, ques_type_val, ans_type_val = my_read_old(
        results_val_old, standard_questions_val_json, standard_annotations_val_json)
    qid_edit_val, pred_ans_edit_val, ss_vc_edit_val, img_ids_edit_val, ques_edit_val, all_ans_edit_val, ques_type_edit_val, ans_type_edit_val  = my_read_old(results_edit_val_old, standard_questions_edit_val_json, standard_annotations_edit_val_json)

    masking_indices_where_qid_edit_but_no_orig_qid = [idx for idx, i in enumerate(qid_edit_val) if i not in qid_val]
    stop_idx = masking_indices_where_qid_edit_but_no_orig_qid[0]
    qid_edit_val = qid_edit_val[0:stop_idx]
    qid_edit_val, pred_ans_edit_val, ss_vc_edit_val, img_ids_edit_val, ques_edit_val, all_ans_edit_val, ques_type_edit_val, ans_type_edit_val = [
        i[0:stop_idx] for i in
        [qid_edit_val, pred_ans_edit_val, ss_vc_edit_val, img_ids_edit_val, ques_edit_val, all_ans_edit_val,
         ques_type_edit_val, ans_type_edit_val]]

    len_val = len(img_ids_val)
    len_edit_val = len(img_ids_edit_val)
    all_indices_val = np.arange(len_val)
    all_indices_edit_val = np.arange(len_edit_val)

    img_ids_val = [str(i).zfill(12) for i in img_ids_val]

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


    qid_gt_ans_label = {}
    for idx, a in enumerate(qid_val):
        qid_gt_ans_label[a] = all_ans_val[idx]
    extended_gt_ans_val = [qid_gt_ans_label[q_id][0] for q_id in qid_edit_val]
    ######new edit
    gt_ans_edit_val = [i[0] for i in all_ans_edit_val]
    gt_ans_val = [i[0] for i in all_ans_val]
    assert [int(gt_ans_edit_val[i]) + 1 == int(extended_gt_ans_val[i]) for i in range(len(gt_ans_edit_val))]
    chuck_many = [i for i, val in enumerate(pred_ans_edit_val) if ans_vocab_list[val] == 'many']
    print('many there in predicted answers, so we would remove them; #many: ', len(chuck_many))

    if len(chuck_many) != 0:
        ### for entire set counting how mnay labels flipped
        labels_flipped_count = np.sum([ans_vocab_list[extended_pred_ans_val[i]] != str(int(ans_vocab_list[val]) + 1) if
                                       ans_vocab_list[val] != 'many' else 0 for i, val in enumerate(pred_ans_edit_val)])
        # labels_flipped_count = np.sum([ans_vocab_list[extended_pred_ans_val[i]] != str(int(ans_vocab_list[val])+1) for i, val in enumerate(pred_ans_edit_val)])
        labels_remained_same_count = np.sum([ans_vocab_list[extended_pred_ans_val[i]] == str(
            int(ans_vocab_list[val]) + 1) if ans_vocab_list[val] != 'many' else 0 for i, val in
                                             enumerate(pred_ans_edit_val)])
        assert (labels_flipped_count + labels_remained_same_count + len(chuck_many) == len(pred_ans_edit_val))

        indices_not_many = [i for i, val in enumerate(pred_ans_edit_val) if ans_vocab_list[val] != 'many']

        lab_fl_ind = [i for i in indices_not_many if ans_vocab_list[extended_pred_ans_val[i]]
                      != str(int(ans_vocab_list[pred_ans_edit_val[i]]) + 1)]
        lab_fl_pos = [i for i in lab_fl_ind if ans_vocab_list[extended_pred_ans_val[i]] != extended_gt_ans_val[i]
                      and ans_vocab_list[pred_ans_edit_val[i]] == gt_ans_edit_val[i]]
        lab_fl_neg = [i for i in lab_fl_ind if ans_vocab_list[extended_pred_ans_val[i]] == extended_gt_ans_val[i]
                      and ans_vocab_list[pred_ans_edit_val[i]] != gt_ans_edit_val[i]]
        lab_fl_right = [i for i in lab_fl_ind if ans_vocab_list[extended_pred_ans_val[i]] == extended_gt_ans_val[i]
                        and ans_vocab_list[pred_ans_edit_val[i]] == gt_ans_edit_val[i]]
        lab_fl_wrong = [i for i in lab_fl_ind if ans_vocab_list[extended_pred_ans_val[i]] != extended_gt_ans_val[i]
                        and ans_vocab_list[pred_ans_edit_val[i]] != gt_ans_edit_val[i]]
        assert (len(lab_fl_pos) + len(lab_fl_neg) + len(lab_fl_right) + len(lab_fl_wrong) == len(lab_fl_ind))

        lab_sm_ind = [i for i in indices_not_many if ans_vocab_list[extended_pred_ans_val[i]]
                      == str(int(ans_vocab_list[pred_ans_edit_val[i]]) + 1)]
        lab_sm_right = [i for i in lab_sm_ind if ans_vocab_list[extended_pred_ans_val[i]] == extended_gt_ans_val[i]
                        and ans_vocab_list[pred_ans_edit_val[i]] == gt_ans_edit_val[i]]
        lab_sm_wrong = [i for i in lab_sm_ind if ans_vocab_list[extended_pred_ans_val[i]] != extended_gt_ans_val[i]
                        and ans_vocab_list[pred_ans_edit_val[i]] != gt_ans_edit_val[i]]
        assert (len(lab_sm_right) + len(lab_sm_wrong) == len(lab_sm_ind))


    else:
        ### for entire set counting how mnay labels flipped
        labels_flipped_count = np.sum(
            [ans_vocab_list[extended_pred_ans_val[i]] != str(int(ans_vocab_list[val]) + 1) for i, val in
             enumerate(pred_ans_edit_val)])
        # labels_flipped_count = np.sum([ans_vocab_list[extended_pred_ans_val[i]] != str(int(ans_vocab_list[val])+1) for i, val in enumerate(pred_ans_edit_val)])
        labels_remained_same_count = np.sum(
            [ans_vocab_list[extended_pred_ans_val[i]] == str(int(ans_vocab_list[val]) + 1) for i, val in
             enumerate(pred_ans_edit_val)])
        assert (labels_flipped_count + labels_remained_same_count == len(pred_ans_edit_val))

        lab_fl_ind = [i for i in range(len_edit_val) if ans_vocab_list[extended_pred_ans_val[i]]
                      != str(int(ans_vocab_list[pred_ans_edit_val[i]]) + 1)]

        lab_fl_pos = [i for i in lab_fl_ind if ans_vocab_list[extended_pred_ans_val[i]] != extended_gt_ans_val[i]
                      and ans_vocab_list[pred_ans_edit_val[i]] == gt_ans_edit_val[i]]
        lab_fl_neg = [i for i in lab_fl_ind if ans_vocab_list[extended_pred_ans_val[i]] == extended_gt_ans_val[i]
                      and ans_vocab_list[pred_ans_edit_val[i]] != gt_ans_edit_val[i]]
        lab_fl_right = [i for i in lab_fl_ind if ans_vocab_list[extended_pred_ans_val[i]] == extended_gt_ans_val[i]
                        and ans_vocab_list[pred_ans_edit_val[i]] == gt_ans_edit_val[i]]
        lab_fl_wrong = [i for i in lab_fl_ind if ans_vocab_list[extended_pred_ans_val[i]] != extended_gt_ans_val[i]
                        and ans_vocab_list[pred_ans_edit_val[i]] != gt_ans_edit_val[i]]
        assert (len(lab_fl_pos) + len(lab_fl_neg) + len(lab_fl_right) + len(lab_fl_wrong) == len(lab_fl_ind))

        lab_sm_ind = [i for i in range(len_edit_val) if ans_vocab_list[extended_pred_ans_val[i]]
                      == str(int(ans_vocab_list[pred_ans_edit_val[i]]) + 1)]
        lab_sm_right = [i for i in lab_sm_ind if ans_vocab_list[extended_pred_ans_val[i]] == extended_gt_ans_val[i]
                        and ans_vocab_list[pred_ans_edit_val[i]] == gt_ans_edit_val[i]]
        lab_sm_wrong = [i for i in lab_sm_ind if ans_vocab_list[extended_pred_ans_val[i]] != extended_gt_ans_val[i]
                        and ans_vocab_list[pred_ans_edit_val[i]] != gt_ans_edit_val[i]]
        assert (len(lab_sm_right) + len(lab_sm_wrong) == len(lab_sm_ind))

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


    #dict_keys(['Baseline', 'ft_orig', 'ft_orig_del', 'ft_orig_edit_del']

    def labels_fixed( lab_fl,  lab_fl_my):
        return (set(lab_fl)- set(lab_fl).intersection(lab_fl_my))

    lab_fl = {}
    for model_name in ['CNN_LSTM', 'SAAA', 'SNMN']:
        if comparison_against_baseline or args.only_baseline_visualization:
            abcd = indices_labels[model_name]['Baseline'][args.label_check]
        else:
            abcd = indices_labels[model_name]['ft_orig'][args.label_check]
            
        if not args.only_baseline_visualization:
            abcd2 = indices_labels[model_name]['ft_orig_del'][args.label_check]
            abcd3 = indices_labels[model_name]['ft_orig_edit_del'][args.label_check]
            lab_fl[model_name]= labels_fixed(abcd, abcd2).union(labels_fixed(abcd, abcd3)) 

        if args.only_baseline_visualization:
            lab_fl[model_name]= set(abcd)


    lab_fl_all31 = set(lab_fl['CNN_LSTM']).intersection( set(lab_fl['SAAA'])) # 37
    lab_fl_all32 = set(lab_fl['SNMN']).intersection(set(lab_fl['SAAA']))  # 17
    lab_fl_all33 = set(lab_fl['CNN_LSTM']).intersection(set(lab_fl['SNMN'])) # 17
    lab_fl_all30 = set(lab_fl['CNN_LSTM']).intersection( set(lab_fl['SAAA']), set(lab_fl['SNMN'])) # 1



    #chosen_indices = [i for i in range(len(img_ids_edit_val))]
    #chosen_diff_list_norm = L_norm_diff
    #chosen_diff_list_label = diff_ss_val_label
    ## chosen_indices, chosen_diff_list_norm, chosen_diff_list_label, plt_title_suffix = get_indices_diff_list_suffix(
    ##     coco_dict_inv, img_ids_edit_val, ques_type_edit_val, ans_type_edit_val, L_norm_diff, diff_ss_val_label, CLASS=None ,QUES_TYPE=None, ANS_TYPE=None, preselected_indices=None, preselected_string=None)
    ##     #preselected_indices=all_3_users_intersection_yes, preselected_string='all 3 annotators agreed')     #TODO  MAKE CHANGES HERE in case you want to look at those indices whch humans agree
    #chosen_color_vector = [color_vector[i] for i in chosen_indices]

    #ipdb.set_trace()
    
    for lab_fl_all3 in [lab_fl_all30 ] : #,  lab_fl_all31, lab_fl_all32, lab_fl_all33]:    
        print(len(lab_fl_all3))
        
        
        
        chosen_indices = list(lab_fl_all3)  #[i for i in range(len(img_ids_edit_val))]
        
        
        if args.answer_not_zero_one:
            chosen_indices = [i for i in chosen_indices if gt_ans_edit_val[i] != '0']
            chosen_indices = [i for i in chosen_indices if gt_ans_edit_val[i] != '1']
            print(len(chosen_indices))
            

        if args.check_spcific_indices_by_img_id:
            img_ids_wanted = [str(316000).zfill(12) + '_' + str(53773).zfill(12), str(131276).zfill(12) + '_' + str(1728537).zfill(12) ,  str(559955).zfill(12) + '_' + str(248899).zfill(12),  str(536110).zfill(12) + '_' + str(44495).zfill(12),
                    str(277498).zfill(12) + '_' + str(591518).zfill(12), str(203063).zfill(12) + '_' + str(63233).zfill(12)]
              
            img_ids_wanted = [str(109907).zfill(12) + '_' + str(8114).zfill(12)]
            chosen_indices= []
            for i in img_ids_wanted:
                for idx, i2 in enumerate(img_ids_edit_val):
                    if i==i2:
                        chosen_indices.append(idx)
            
            #chosen_indices = [img_ids_edit_val.index(i) for i in img_ids_wanted if i in img_ids_edit_val]
            

            
            
        chosen_diff_list_norm = [L_norm_diff[i] for i in chosen_indices]
        chosen_diff_list_label = [diff_ss_val_label[i] for i in chosen_indices]
        chosen_color_vector = [color_vector[i] for i in chosen_indices]
        
        
        def onpick(event):

            index = event.ind
            print('event_ind:', index)  ## event_ind: [1394 1419] ## sensitive to the plot- so ind: 0 to len(whatever you are plotting)

            # index: 1567.8072383626895
            # event_ind: [1544 1546 1550 1551 1552 1555 1557 1558 1559 1561 1562 1563 1564 1566 ...]


            ind = chosen_indices[index[0]]                #TODO vedika so whatever is plotted: all_indices/ chosen_indices/lab_fl_neg/lab_fl_pos...and so on
            question_str = ques_edit_val[ind]

            gt_answers = all_ans_edit_val[ind]
            
            
            # CNN_LSTM
            pred_ans_before_edit = ans_vocab_list[indices_labels['CNN_LSTM']['Baseline']['extended_pred_ans_val'][ind]]
            pred_ans_after_edit = ans_vocab_list[indices_labels['CNN_LSTM']['Baseline']['pred_ans_edit_val'][ind]]
            conf_before = indices_labels['CNN_LSTM']['Baseline']['ext_ss_max_value_val'][ind]
            conf_after = indices_labels['CNN_LSTM']['Baseline']['ss_value_at_val_edit_val'][ind]
            ss_vc_edit_val = indices_labels['CNN_LSTM']['Baseline']['ss_vc_edit_val']
            diff_ss_val_label = indices_labels['CNN_LSTM']['Baseline']['diff_ss_val_label']


            if not args.only_baseline_visualization:
                pred_ans_before_edit_ft_orig = ans_vocab_list[indices_labels['CNN_LSTM']['ft_orig']['extended_pred_ans_val'][ind]]
                pred_ans_after_edit_ft_orig = ans_vocab_list[indices_labels['CNN_LSTM']['ft_orig']['pred_ans_edit_val'][ind]]
                conf_before_ft_orig = indices_labels['CNN_LSTM']['ft_orig']['ext_ss_max_value_val'][ind]
                conf_after_ft_orig = indices_labels['CNN_LSTM']['ft_orig']['ss_value_at_val_edit_val'][ind]
                ss_vc_edit_val_ft_orig = indices_labels['CNN_LSTM']['ft_orig']['ss_vc_edit_val']
                diff_ss_val_label_ft_orig = indices_labels['CNN_LSTM']['ft_orig']['diff_ss_val_label']
                
                
                pred_ans_before_edit_ft_del = ans_vocab_list[indices_labels['CNN_LSTM']['ft_orig_del']['extended_pred_ans_val'][ind]]
                pred_ans_after_edit_ft_del = ans_vocab_list[indices_labels['CNN_LSTM']['ft_orig_del']['pred_ans_edit_val'][ind]]
                conf_before_ft_del = indices_labels['CNN_LSTM']['ft_orig_del']['ext_ss_max_value_val'][ind]
                conf_after_ft_del = indices_labels['CNN_LSTM']['ft_orig_del']['ss_value_at_val_edit_val'][ind]
                ss_vc_edit_val_ft_del = indices_labels['CNN_LSTM']['ft_orig_del']['ss_vc_edit_val']
                diff_ss_val_label_ft_del = indices_labels['CNN_LSTM']['ft_orig_del']['diff_ss_val_label']


                pred_ans_before_edit_ft_edit_del = ans_vocab_list[indices_labels['CNN_LSTM']['ft_orig_edit_del']['extended_pred_ans_val'][ind]]
                pred_ans_after_edit_ft_edit_del = ans_vocab_list[indices_labels['CNN_LSTM']['ft_orig_edit_del']['pred_ans_edit_val'][ind]]
                conf_before_ft_edit_del = indices_labels['CNN_LSTM']['ft_orig_edit_del']['ext_ss_max_value_val'][ind]
                conf_after_ft_edit_del = indices_labels['CNN_LSTM']['ft_orig_edit_del']['ss_value_at_val_edit_val'][ind]
                ss_vc_edit_val_ft_edit_del = indices_labels['CNN_LSTM']['ft_orig_edit_del']['ss_vc_edit_val']
                diff_ss_val_label_ft_edit_del = indices_labels['CNN_LSTM']['ft_orig_edit_del']['diff_ss_val_label']


            # SAAA
            pred_ans_before_edit1 = ans_vocab_list1[indices_labels['SAAA']['Baseline']['extended_pred_ans_val'][ind]]
            pred_ans_after_edit1 = ans_vocab_list1[indices_labels['SAAA']['Baseline']['pred_ans_edit_val'][ind]]
            conf_before1= indices_labels['SAAA']['Baseline']['ext_ss_max_value_val'][ind]
            conf_after1 = indices_labels['SAAA']['Baseline']['ss_value_at_val_edit_val'][ind]
            ss_vc_edit_val1 = indices_labels['SAAA']['Baseline']['ss_vc_edit_val']
            diff_ss_val_label1 =indices_labels['SAAA']['Baseline']['diff_ss_val_label']

            if not args.only_baseline_visualization:
                pred_ans_before_edit1_ft_orig = ans_vocab_list1[indices_labels['SAAA']['ft_orig']['extended_pred_ans_val'][ind]]
                pred_ans_after_edit1_ft_orig = ans_vocab_list1[indices_labels['SAAA']['ft_orig']['pred_ans_edit_val'][ind]]
                conf_before1_ft_orig = indices_labels['SAAA']['ft_orig']['ext_ss_max_value_val'][ind]
                conf_after1_ft_orig = indices_labels['SAAA']['ft_orig']['ss_value_at_val_edit_val'][ind]
                ss_vc_edit_val1_ft_orig = indices_labels['SAAA']['ft_orig']['ss_vc_edit_val']
                diff_ss_val_label1_ft_orig = indices_labels['SAAA']['ft_orig']['diff_ss_val_label']
                
                
                pred_ans_before_edit1_ft_del = ans_vocab_list1[indices_labels['SAAA']['ft_orig_del']['extended_pred_ans_val'][ind]]
                pred_ans_after_edit1_ft_del = ans_vocab_list1[indices_labels['SAAA']['ft_orig_del']['pred_ans_edit_val'][ind]]
                conf_before1_ft_del = indices_labels['SAAA']['ft_orig_del']['ext_ss_max_value_val'][ind]
                conf_after1_ft_del = indices_labels['SAAA']['ft_orig_del']['ss_value_at_val_edit_val'][ind]
                ss_vc_edit_val1_ft_del = indices_labels['SAAA']['ft_orig_del']['ss_vc_edit_val']
                diff_ss_val_label1_ft_del = indices_labels['SAAA']['ft_orig_del']['diff_ss_val_label']
                
                
                pred_ans_before_edit1_ft_edit_del = ans_vocab_list1[indices_labels['SAAA']['ft_orig_edit_del']['extended_pred_ans_val'][ind]]
                pred_ans_after_edit1_ft_edit_del = ans_vocab_list1[indices_labels['SAAA']['ft_orig_edit_del']['pred_ans_edit_val'][ind]]
                conf_before1_ft_edit_del = indices_labels['SAAA']['ft_orig_edit_del']['ext_ss_max_value_val'][ind]
                conf_after1_ft_edit_del = indices_labels['SAAA']['ft_orig_edit_del']['ss_value_at_val_edit_val'][ind]
                ss_vc_edit_val1_ft_edit_del = indices_labels['SAAA']['ft_orig_edit_del']['ss_vc_edit_val']
                diff_ss_val_label1_ft_edit_del = indices_labels['SAAA']['ft_orig_edit_del']['diff_ss_val_label']

            # SNMN
            pred_ans_before_edit2 = ans_vocab_list2[indices_labels['SNMN']['Baseline']['extended_pred_ans_val'][ind]]
            pred_ans_after_edit2 = ans_vocab_list2[indices_labels['SNMN']['Baseline']['pred_ans_edit_val'][ind]]
            conf_before2 = indices_labels['SNMN']['Baseline']['ext_ss_max_value_val'][ind]
            conf_after2 = indices_labels['SNMN']['Baseline']['ss_value_at_val_edit_val'][ind]
            ss_vc_edit_val2 = indices_labels['SNMN']['Baseline']['ss_vc_edit_val']
            diff_ss_val_label2  = indices_labels['SNMN']['Baseline']['diff_ss_val_label']


            if not args.only_baseline_visualization:
                pred_ans_before_edit2_ft_orig = ans_vocab_list2[indices_labels['SNMN']['ft_orig']['extended_pred_ans_val'][ind]]
                pred_ans_after_edit2_ft_orig = ans_vocab_list2[indices_labels['SNMN']['ft_orig']['pred_ans_edit_val'][ind]]
                conf_before2_ft_orig = indices_labels['SNMN']['ft_orig']['ext_ss_max_value_val'][ind]
                conf_after2_ft_orig = indices_labels['SNMN']['ft_orig']['ss_value_at_val_edit_val'][ind]
                ss_vc_edit_val2_ft_orig = indices_labels['SNMN']['ft_orig']['ss_vc_edit_val']
                diff_ss_val_label2_ft_orig = indices_labels['SNMN']['ft_orig']['diff_ss_val_label']
                
                
                pred_ans_before_edit2_ft_del = ans_vocab_list2[indices_labels['SNMN']['ft_orig_del']['extended_pred_ans_val'][ind]]
                pred_ans_after_edit2_ft_del = ans_vocab_list2[indices_labels['SNMN']['ft_orig_del']['pred_ans_edit_val'][ind]]
                conf_before2_ft_del = indices_labels['SNMN']['ft_orig_del']['ext_ss_max_value_val'][ind]
                conf_after2_ft_del = indices_labels['SNMN']['ft_orig_del']['ss_value_at_val_edit_val'][ind]
                ss_vc_edit_val2_ft_del = indices_labels['SNMN']['ft_orig_del']['ss_vc_edit_val']
                diff_ss_val_label2_ft_del = indices_labels['SNMN']['ft_orig_del']['diff_ss_val_label']
                    

                pred_ans_before_edit2_ft_edit_del = ans_vocab_list2[indices_labels['SNMN']['ft_orig_edit_del']['extended_pred_ans_val'][ind]]
                pred_ans_after_edit2_ft_edit_del = ans_vocab_list2[indices_labels['SNMN']['ft_orig_edit_del']['pred_ans_edit_val'][ind]]
                conf_before2_ft_edit_del = indices_labels['SNMN']['ft_orig_edit_del']['ext_ss_max_value_val'][ind]
                conf_after2_ft_edit_del = indices_labels['SNMN']['ft_orig_edit_del']['ss_value_at_val_edit_val'][ind]
                ss_vc_edit_val2_ft_edit_del = indices_labels['SNMN']['ft_orig_edit_del']['ss_vc_edit_val']
                diff_ss_val_label2_ft_edit_del = indices_labels['SNMN']['ft_orig_edit_del']['diff_ss_val_label']


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
            vcat = cv2.copyMakeBorder(img1, 195, 10, 50,50, cv2.BORDER_CONSTANT, value=WHITE)   ## 100,100
            #cv2.imshow('constant', vcat)
            #white = np.zeros((250, 600, 3), np.uint8)  ## img1.shape[1]
            #white[:] = (255, 255, 255)
            #vcat = cv2.vconcat((white, img1))
            # if img_ids_edit_val[ind] == '000000147921_000000000016' and qid_edit_val[ind] == '147921006':
            #     ipdb.set_trace()


            model1 = 'SAAA'
            model2= 'SNMN'
                
            str1 = str(question_str) +  '   image_id:' + str(img_ids_edit_val[ind]) + 'ques_id: ' + str(qid_edit_val[ind])
            str2 = 'model_Baseline' + '       ans_bef' + '    ans_aft' + '  conf aft_ans_before' +  '  diff_label_conf'

            str3 = model0 +  '   {}:'.format(pred_ans_before_edit) + str(round(conf_before,3)) + '   ' + \
                        '   {}:'.format(pred_ans_after_edit) + str(round(np.max(ss_vc_edit_val[ind]),3)) + \
                        '   {}:'.format(pred_ans_before_edit) + str(round(conf_after,3))  + \
                        '    ' + str(round(diff_ss_val_label[ind], 3))

            str4 = model1  + '   {}:'.format(pred_ans_before_edit1) + str(round(conf_before1,3)) + '   ' + \
                        '   {}:'.format(pred_ans_after_edit1) + str(round(np.max(ss_vc_edit_val1[ind]),3)) + \
                        '   {}:'.format(pred_ans_before_edit1) + str(round(conf_after1,3))  + \
                        '    ' + str(round(diff_ss_val_label1[ind], 3))

            str5 = model2  + '   {}:'.format(pred_ans_before_edit2) + str(round(conf_before2,3)) + '   ' + \
                        '   {}:'.format(pred_ans_after_edit2) + str(round(np.max(ss_vc_edit_val2[ind]),3)) + \
                        '   {}:'.format(pred_ans_before_edit2) + str(round(conf_after2,3))  + \
                        '    ' + str(round(diff_ss_val_label2[ind], 3))

            str6 = 'all_gt_answers: ' + str(gt_answers)


            if not args.only_baseline_visualization:

                
                str21 = 'model_ft_orig' + '       ans_bef' + '    ans_aft' + '  conf aft_ans_before' +  '  diff_label_conf'

                str31 = model0 +  '   {}:'.format(pred_ans_before_edit_ft_orig) + str(round(conf_before_ft_orig,3)) + '   ' + \
                        '   {}:'.format(pred_ans_after_edit_ft_orig) + str(round(np.max(ss_vc_edit_val_ft_orig[ind]),3)) + \
                        '   {}:'.format(pred_ans_before_edit_ft_orig) + str(round(conf_after_ft_orig,3))  + \
                        '    ' + str(round(diff_ss_val_label_ft_orig[ind], 3))

                str41 = model1  + '   {}:'.format(pred_ans_before_edit1_ft_orig) + str(round(conf_before1_ft_orig,3)) + '   ' + \
                        '   {}:'.format(pred_ans_after_edit1_ft_orig) + str(round(np.max(ss_vc_edit_val1_ft_orig[ind]),3)) + \
                        '   {}:'.format(pred_ans_before_edit1_ft_orig) + str(round(conf_after1_ft_orig,3))  + \
                        '    ' + str(round(diff_ss_val_label1_ft_orig[ind], 3))

                str51 = model2  + '   {}:'.format(pred_ans_before_edit2_ft_orig) + str(round(conf_before2_ft_orig,3)) + '   ' + \
                        '   {}:'.format(pred_ans_after_edit2_ft_orig) + str(round(np.max(ss_vc_edit_val2_ft_orig[ind]),3)) + \
                        '   {}:'.format(pred_ans_before_edit2_ft_orig) + str(round(conf_after2_ft_orig,3))  + \
                        '    ' + str(round(diff_ss_val_label2_ft_orig[ind], 3))




                str22 = 'model_ft_orig_del' + '       ans_bef' + '    ans_aft' + '  conf aft_ans_before' +  '  diff_label_conf'

                str32 = model0 +  '   {}:'.format(pred_ans_before_edit_ft_del) + str(round(conf_before_ft_del,3)) + '   ' + \
                        '   {}:'.format(pred_ans_after_edit_ft_del) + str(round(np.max(ss_vc_edit_val_ft_del[ind]),3)) + \
                        '   {}:'.format(pred_ans_before_edit_ft_del) + str(round(conf_after_ft_del,3))  + \
                        '    ' + str(round(diff_ss_val_label_ft_del[ind], 3))

                str42 = model1  + '   {}:'.format(pred_ans_before_edit1_ft_del) + str(round(conf_before1_ft_del,3)) + '   ' + \
                        '   {}:'.format(pred_ans_after_edit1_ft_del) + str(round(np.max(ss_vc_edit_val1_ft_del[ind]),3)) + \
                        '   {}:'.format(pred_ans_before_edit1_ft_del) + str(round(conf_after1_ft_del,3))  + \
                        '    ' + str(round(diff_ss_val_label1_ft_del[ind], 3))

                str52 = model2  + '   {}:'.format(pred_ans_before_edit2_ft_del) + str(round(conf_before2_ft_del,3)) + '   ' + \
                        '   {}:'.format(pred_ans_after_edit2_ft_del) + str(round(np.max(ss_vc_edit_val2_ft_del[ind]),3)) + \
                        '   {}:'.format(pred_ans_before_edit2_ft_del) + str(round(conf_after2_ft_del,3))  + \
                        '    ' + str(round(diff_ss_val_label2_ft_del[ind], 3))



                str23 = 'model_ft_orig_edit_del' + '       ans_bef' + '    ans_aft' + '  conf aft_ans_before' +  '  diff_label_conf'

                str33 = model0 +  '   {}:'.format(pred_ans_before_edit_ft_edit_del) + str(round(conf_before_ft_edit_del,3)) + '   ' + \
                        '   {}:'.format(pred_ans_after_edit_ft_edit_del) + str(round(np.max(ss_vc_edit_val_ft_edit_del[ind]),3)) + \
                        '   {}:'.format(pred_ans_before_edit_ft_edit_del) + str(round(conf_after_ft_edit_del,3))  + \
                        '    ' + str(round(diff_ss_val_label_ft_edit_del[ind], 3))

                str43 = model1  + '   {}:'.format(pred_ans_before_edit1_ft_edit_del) + str(round(conf_before1_ft_edit_del,3)) + '   ' + \
                        '   {}:'.format(pred_ans_after_edit1_ft_edit_del) + str(round(np.max(ss_vc_edit_val1_ft_edit_del[ind]),3)) + \
                        '   {}:'.format(pred_ans_before_edit1_ft_edit_del) + str(round(conf_after1_ft_edit_del,3))  + \
                        '    ' + str(round(diff_ss_val_label1_ft_edit_del[ind], 3))

                str53 = model2  + '   {}:'.format(pred_ans_before_edit2_ft_edit_del) + str(round(conf_before2_ft_edit_del,3)) + '   ' + \
                        '   {}:'.format(pred_ans_after_edit2_ft_edit_del) + str(round(np.max(ss_vc_edit_val2_ft_edit_del[ind]),3)) + \
                        '   {}:'.format(pred_ans_before_edit2_ft_edit_del) + str(round(conf_after2_ft_edit_del,3))  + \
                        '    ' + str(round(diff_ss_val_label2_ft_edit_del[ind], 3))

                vcat = cv2.copyMakeBorder(img1, 520, 10, 50,50, cv2.BORDER_CONSTANT, value=WHITE)   ## 100,100 #180
            
            font_sz = 0.50
            cv2.putText(vcat, str1, (30,25), cv2.FONT_HERSHEY_SIMPLEX, font_sz, (0,0,0), 1, cv2.LINE_AA)
            cv2.putText(vcat, str6, (30, 50), cv2.FONT_HERSHEY_SIMPLEX, font_sz, (0, 0, 0), 1, cv2.LINE_AA)

            cv2.putText(vcat, '---------------------------------------------------------------------------', (30,75), cv2.FONT_HERSHEY_SIMPLEX, font_sz, (0, 0, 0), 1, cv2.LINE_AA)

            cv2.putText(vcat, str2, (30,90), cv2.FONT_HERSHEY_SIMPLEX, font_sz, (0, 0, 0), 1, cv2.LINE_AA)
            cv2.putText(vcat, str3, (30,115), cv2.FONT_HERSHEY_SIMPLEX, font_sz, (0, 0, 0), 1, cv2.LINE_AA)
            cv2.putText(vcat, str4, (30,140), cv2.FONT_HERSHEY_SIMPLEX, font_sz, (0, 0, 0), 1, cv2.LINE_AA)
            cv2.putText(vcat, str5, (30,165), cv2.FONT_HERSHEY_SIMPLEX, font_sz, (0, 0, 0), 1, cv2.LINE_AA)

            if not args.only_baseline_visualization:

                
                cv2.putText(vcat, '---------------------------------------------------------------------------', (30,190), cv2.FONT_HERSHEY_SIMPLEX, font_sz, (0, 0, 0), 1, cv2.LINE_AA)


                cv2.putText(vcat, str21, (30,200), cv2.FONT_HERSHEY_SIMPLEX, font_sz, (0, 0, 0), 1, cv2.LINE_AA)
                cv2.putText(vcat, str31, (30,225), cv2.FONT_HERSHEY_SIMPLEX, font_sz, (0, 0, 0), 1, cv2.LINE_AA)
                cv2.putText(vcat, str41, (30,250), cv2.FONT_HERSHEY_SIMPLEX, font_sz, (0, 0, 0), 1, cv2.LINE_AA)
                cv2.putText(vcat, str51, (30,275), cv2.FONT_HERSHEY_SIMPLEX, font_sz, (0, 0, 0), 1, cv2.LINE_AA)

                cv2.putText(vcat, '---------------------------------------------------------------------------', (30,300), cv2.FONT_HERSHEY_SIMPLEX, font_sz, (0, 0, 0), 1, cv2.LINE_AA)

                cv2.putText(vcat, str22, (30,310), cv2.FONT_HERSHEY_SIMPLEX, font_sz, (0, 0, 0), 1, cv2.LINE_AA)
                cv2.putText(vcat, str32, (30,335), cv2.FONT_HERSHEY_SIMPLEX, font_sz, (0, 0, 0), 1, cv2.LINE_AA)
                cv2.putText(vcat, str42, (30,360), cv2.FONT_HERSHEY_SIMPLEX, font_sz, (0, 0, 0), 1, cv2.LINE_AA)
                cv2.putText(vcat, str52, (30,385), cv2.FONT_HERSHEY_SIMPLEX, font_sz, (0, 0, 0), 1, cv2.LINE_AA)
                
                cv2.putText(vcat, '---------------------------------------------------------------------------', (30,310), cv2.FONT_HERSHEY_SIMPLEX, font_sz, (0, 0, 0), 1, cv2.LINE_AA)

                cv2.putText(vcat, str23, (30,420), cv2.FONT_HERSHEY_SIMPLEX, font_sz, (0, 0, 0), 1, cv2.LINE_AA)
                cv2.putText(vcat, str33, (30,445), cv2.FONT_HERSHEY_SIMPLEX, font_sz, (0, 0, 0), 1, cv2.LINE_AA)
                cv2.putText(vcat, str43, (30,470), cv2.FONT_HERSHEY_SIMPLEX, font_sz, (0, 0, 0), 1, cv2.LINE_AA)
                cv2.putText(vcat, str53, (30,495), cv2.FONT_HERSHEY_SIMPLEX, font_sz, (0, 0, 0), 1, cv2.LINE_AA)

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
        ax.set_title('click to see, diff_in_softmax_value: {}'.format('counting_del1'))
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
