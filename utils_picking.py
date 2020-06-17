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
import random



def my_read_old(results_old_pkl, standard_q_json, standard_a_json=None):
    st = time.time()
    with open(results_old_pkl, 'rb') as file:
        res_val = pickle.load(file)
    if str(standard_q_json)[-4:] == 'json':
        with open(standard_q_json) as file:
            details_ques = json.load(file)['questions']
    else:
        with open(standard_q_json, 'rb') as file:
            details_ques = pickle.load(file)['questions']

    ## from results file- model specific
    pred_ans = [details['ans_id'] for details in res_val]
    softmax_vector = [details['ss_vc'] for details in res_val]
    img_id_res = [details['img_id'] for details in res_val]
    qid_res = [details['ques_id'] for details in res_val]
    # gt_ans_used = [details['gt_ans_id_used'] for details in res_val] ## in case of snmn
    # q_ids = [details['ques_id'] for details in res_val]     ### in cases of snmn
    # assert (q_ids == [details_q['question_id'] for details_q in details_ques])

    q_ids = [details_q['question_id'] for details_q in details_ques]
    img_ids = [details_q['image_id'] for details_q in details_ques]
    #img_ids = [str(iid).zfill(12) for iid in img_ids]   ## TODO : latest addition - to handle changes- now all image_ids are string!!!
    ques_str = [details_q['question'] for details_q in details_ques]

    ### make sure order of qid, img_id consistent between results and standard annotations files
    ## if not, sort them to what is there in the standard files
    #ipdb.set_trace()
    if qid_res != q_ids and img_id_res != img_ids:
        std_q_img_id = {}

        ### SAA and CNNN_LSTM in case for train set: we find a discrpancy so qids are nto a strict subset of qid_res
        if len((set(q_ids) & set(qid_res))) != len(set(q_ids)):
            ipdb.set_trace()
            print()
            print(' DISCREPANCY ALERT- SIMILAR TO WHAT YOU SAW IN CASE OF SAA/CNN_LSTM WHEN YOU WERE TESTING ON TRAIN  ')
            print()
            target_q_ids = list((set(q_ids) & set(qid_res)))
        else:    ## ideally q_ids is a strict subset of res_qids...
            target_q_ids = q_ids

        for qid_idx, qid in enumerate(target_q_ids):
            key = str(qid) +  str(img_ids[qid_idx])
            std_q_img_id[key] = qid_idx

        res_q_img_id_pans_ss_vc = {}
        for qid_idx, qid in enumerate(qid_res):
            # if 'torch' in img_id_res[qid_idx].type():
            #     img_id_res[qid_idx] = img_id_res[qid_idx].item()
            # if len(str(img_id_res[qid_idx])) < 20:  ## SAAA fix; for edit_set- img_id is string and not torch object
            #     img_id_res[qid_idx] = img_id_res[qid_idx].item()
            key_new = str(qid) + str(img_id_res[qid_idx])
            res_q_img_id_pans_ss_vc[key_new] = pred_ans[qid_idx], softmax_vector[qid_idx], qid, img_id_res[qid_idx]


        pred_ans_corr = [res_q_img_id_pans_ss_vc[key][0] for key in std_q_img_id.keys()]
        softmax_vector_corr = [res_q_img_id_pans_ss_vc[key][1] for key in std_q_img_id.keys()]
        qid_corr = [res_q_img_id_pans_ss_vc[key][2] for key in std_q_img_id.keys()]
        img_id_corr = [res_q_img_id_pans_ss_vc[key][3] for key in std_q_img_id.keys()]
        assert img_id_corr == img_ids
        assert qid_corr == q_ids

        pred_ans = pred_ans_corr
        softmax_vector = softmax_vector_corr


    if standard_a_json is not None:
        with open(standard_a_json) as file:
            details_ann = json.load(file)['annotations']
        all_answers = [[ans['answer'] for ans in details_ann[i]['answers']] for i in range(len(img_ids))]
        # most_freq_ans = [details_ann[i]['multiple_choice_answer'] for i in range(len(img_ids))]
        ques_type_data = [details_ann[i]['question_type'] for i in range(len(img_ids))]
        ans_type_data = [details_ann[i]['answer_type'] for i in range(len(img_ids))]
        print(time.time() - st)
        return q_ids, pred_ans, softmax_vector, img_ids, ques_str, all_answers, ques_type_data, ans_type_data     # ,gt_ans_used
    else:
        print(time.time() - st)
        return q_ids, pred_ans, softmax_vector, img_ids, ques_str, [], [], []  # ,gt_ans_used


def my_read_old_val_90_10(results_old_pkl, standard_q_json, standard_a_json , if_90_10=0):
    st = time.time()
    with open(results_old_pkl, 'rb') as file:
        res_val = pickle.load(file)
    if str(standard_q_json)[-4:] == 'json':
        with open(standard_q_json) as file:
            details_ques_all = json.load(file)['questions']
            len_90 = int(0.9*len(details_ques_all))
            details_ques = details_ques_all[0:len_90]
    else:
        with open(standard_q_json, 'rb') as file:
            details_ques_all = pickle.load(file)['questions']
            len_90 = int(0.9*len(details_ques_all))
            details_ques = details_ques_all[0:len_90]


    q_ids_q = [details_q['question_id'] for details_q in details_ques]
    img_ids_q = [details_q['image_id'] for details_q in details_ques]
    ques_str_q = [details_q['question'] for details_q in details_ques]


    with open(standard_a_json) as file:
        details_ann = json.load(file)['annotations']
    all_answers = [[ans['answer'] for ans in details_ann[i]['answers']] for i in range(len(img_ids_q))]
    # most_freq_ans = [details_ann[i]['multiple_choice_answer'] for i in range(len(img_ids))]
    ques_type_data = [details_ann[i]['question_type'] for i in range(len(img_ids_q))]
    ans_type_data = [details_ann[i]['answer_type'] for i in range(len(img_ids_q))]

    q_ids = [details_q['question_id'] for i,details_q in enumerate(details_ques)]
    img_ids = [details_q['image_id'] for i,details_q in enumerate(details_ques)]
    ques_str = [details_q['question'] for i,details_q in enumerate(details_ques)]


    if if_90_10==1:
        q_ids = [details_q['question_id'] for i,details_q in enumerate(details_ques) if len(set(all_answers[i]))==1]
        img_ids = [details_q['image_id'] for i,details_q in enumerate(details_ques) if len(set(all_answers[i]))==1]
        ques_str = [details_q['question'] for i,details_q in enumerate(details_ques) if len(set(all_answers[i]))==1]
        ques_type_data = [details_ann[i]['question_type'] for i in range(len(img_ids_q)) if len(set(all_answers[i]))==1]
        ans_type_data = [details_ann[i]['answer_type'] for i in range(len(img_ids_q)) if len(set(all_answers[i]))==1]


    ## from results file- model specific
    pred_ans = [details['ans_id'] for details in res_val]
    softmax_vector = [details['ss_vc'] for details in res_val]
    img_id_res = [details['img_id'] for details in res_val]
    qid_res = [details['ques_id'] for details in res_val]
    # gt_ans_used = [details['gt_ans_id_used'] for details in res_val] ## in case of snmn
    # q_ids = [details['ques_id'] for details in res_val]     ### in cases of snmn
    # assert (q_ids == [details_q['question_id'] for details_q in details_ques])



    ### make sure order of qid, img_id consistent between results and standard annotations files
    ## if not, sort them to what is there in the standard files
    # ipdb.set_trace()
    if qid_res != q_ids and img_id_res != img_ids:
        std_q_img_id = {}

        ### SAA and CNNN_LSTM in case for train set: we find a discrpancy so qids are nto a strict subset of qid_res
        if len((set(q_ids) & set(qid_res))) != len(set(q_ids)):
            print()
            print(
                ' DISCREPANCY ALERT- SIMILAR TO WHAT YOU SAW IN CASE OF SAA/CNN_LSTM WHEN YOU WERE TESTING ON TRAIN  ')
            print()
            target_q_ids = list((set(q_ids) & set(qid_res)))
        else:  ## ideally q_ids is a strict subset of res_qids...
            target_q_ids = q_ids

        for qid_idx, qid in enumerate(target_q_ids):
            key = str(qid) + str(img_ids[qid_idx])
            std_q_img_id[key] = qid_idx

        res_q_img_id_pans_ss_vc = {}
        for qid_idx, qid in enumerate(qid_res):
            # if 'torch' in img_id_res[qid_idx].type():
            #     img_id_res[qid_idx] = img_id_res[qid_idx].item()
            # if len(str(img_id_res[qid_idx])) < 20:  ## SAAA fix; for edit_set- img_id is string and not torch object
            #     img_id_res[qid_idx] = img_id_res[qid_idx].item()
            key_new = str(qid) + str(img_id_res[qid_idx])
            res_q_img_id_pans_ss_vc[key_new] = pred_ans[qid_idx], softmax_vector[qid_idx], qid, img_id_res[qid_idx]

        pred_ans_corr = [res_q_img_id_pans_ss_vc[key][0] for key in std_q_img_id.keys()]
        softmax_vector_corr = [res_q_img_id_pans_ss_vc[key][1] for key in std_q_img_id.keys()]
        qid_corr = [res_q_img_id_pans_ss_vc[key][2] for key in std_q_img_id.keys()]
        img_id_corr = [res_q_img_id_pans_ss_vc[key][3] for key in std_q_img_id.keys()]
        assert img_id_corr == img_ids
        assert qid_corr == q_ids

        pred_ans = pred_ans_corr
        softmax_vector = softmax_vector_corr

        print(time.time() - st)
    return q_ids, pred_ans, softmax_vector, img_ids, ques_str, all_answers, ques_type_data, ans_type_data  # ,gt_ans_used


# def my_read_hack(results_old_pkl, standard_q_json, standard_a_json):
#     st = time.time()
#     with open(results_old_pkl, 'rb') as file:
#         res_val = pickle.load(file)
#     if str(standard_q_json)[-4:] == 'json':
#         with open(standard_q_json) as file:
#             details_ques = json.load(file)['questions']
#     else:
#         with open(standard_q_json, 'rb') as file:
#             details_ques = pickle.load(file)['questions']
#
#     ## from results file- model specific
#     pred_ans = [details['ans_id'] for details in res_val]
#     softmax_vector = [details['ss_vc'] for details in res_val]
#     img_id_res = [details['img_id'] for details in res_val]
#     qid_res = [details['ques_id'] for details in res_val]
#     # gt_ans_used = [details['gt_ans_id_used'] for details in res_val] ## in case of snmn
#     # q_ids = [details['ques_id'] for details in res_val]     ### in cases of snmn
#     # assert (q_ids == [details_q['question_id'] for details_q in details_ques])
#
#     q_ids = [details_q['question_id'] for details_q in details_ques]
#     img_ids = [details_q['image_id'] for details_q in details_ques]
#     ques_str = [details_q['question'] for details_q in details_ques]
#
#     with open(standard_a_json) as file:
#         details_ann = json.load(file)['annotations']
#     all_answers = [[ans['answer'] for ans in details_ann[i]['answers']] for i in range(len(img_ids))]
#     # most_freq_ans = [details_ann[i]['multiple_choice_answer'] for i in range(len(img_ids))]
#     ques_type_data = [details_ann[i]['question_type'] for i in range(len(img_ids))]
#     ans_type_data = [details_ann[i]['answer_type'] for i in range(len(img_ids))]
#
#
#     #q_ids, pred_ans, softmax_vector, img_ids, ques_str, all_answers, ques_type_data, ans_type_data
#
#     q_ids_problem_solving = {}
#     for idx,qid in enumerate(q_ids):
#         q_ids_problem_solving[qid] = qid, img_ids[idx], ques_str[idx], all_answers[idx], ques_type_data[idx], ans_type_data[idx]
#
#     q_ids_problem_solving_res = {}
#     for idx,qid in enumerate(qid_res):
#         q_ids_problem_solving_res[qid] = qid, img_id_res[idx], softmax_vector[idx], pred_ans[idx]
#
#     ### make sure order of qid, img_id consistent between results and standard annotations files
#     ## if not, sort them to what is there in the standar files
#
#     if qid_res != q_ids and img_id_res != img_ids:
#         ### SAA and CNNN_LSTM in case for train set: we find a discrpancy so qids are nto a strict subset of qid_res
#         if len((set(q_ids) & set(qid_res))) != len(set(q_ids)): #and len((set(img_ids) & set(img_id_res))) != len(set(img_ids)):
#             print()
#             print(' DISCREPANCY ALERT- FIRST_STAGE: q_ids: SIMILAR TO WHAT YOU SAW IN CASE OF SAA/CNN_LSTM WHEN YOU WERE TESTING ON TRAIN  ')
#             target_q_ids = set(q_ids) & set(qid_res)
#             print('THE LENGTH AT WHICH WE ARE LOOKIG AT: ', len(target_q_ids), 'AS OPPOSED TO ACTUAL: ', len(q_ids))
#             print()
#
#         if len((set(img_ids) & set(img_id_res))) != len(set(img_ids)):
#             print()
#             print(' DISCREPANCY ALERT- SECOND_STAGE: img_ids: SIMILAR TO WHAT YOU SAW IN CASE OF SAA/CNN_LSTM WHEN YOU WERE TESTING ON TRAIN  ')
#             target_img_ids = set(img_ids) & set(img_id_res)
#             print('THE LENGTH AT WHICH WE ARE LOOKIG AT: ', len(target_img_ids), 'AS OPPOSED TO ACTUAL: ', len(img_ids))
#             print()
#
#
#         std_q_img_id = {}
#         res_q_img_id_pans_ss_vc = {}
#         for qid_idx, qid in enumerate(target_q_ids):
#             key = str(qid) +  '_' + str(q_ids_problem_solving[qid][1]) #+  str(img_ids[qid_idx])
#             std_q_img_id[key] = qid_idx
#             key_new = str(qid) + '_' + str(q_ids_problem_solving_res[qid][1])
#             res_q_img_id_pans_ss_vc[key_new] = q_ids_problem_solving_res[qid][3], q_ids_problem_solving_res[qid][2], qid, q_ids_problem_solving_res[qid][1]
#
#         target_keys = set(res_q_img_id_pans_ss_vc.keys()) & set(std_q_img_id.keys())
#         target_keys = list(target_keys)
#
#
#         if len(target_keys[0]) > 25:
#             target_q_ids_from_keys = [int(i.rsplit('_', 2)[0]) for i in target_keys] #[i[:-26] for i in target_keys]
#         else:
#             target_q_ids_from_keys = [int(i.rsplit('_', 1)[0]) for i in target_keys]
#
#
#         pred_ans_corr = [res_q_img_id_pans_ss_vc[key][0] for key in target_keys]#std_q_img_id.keys()]
#         softmax_vector_corr = [res_q_img_id_pans_ss_vc[key][1] for key in target_keys]
#         qid_corr = [res_q_img_id_pans_ss_vc[key][2] for key in target_keys]
#         img_id_corr = [res_q_img_id_pans_ss_vc[key][3] for key in target_keys]
#
#
#         img_id_corr2 = [q_ids_problem_solving[key][1] for key in target_q_ids_from_keys] #target_q_ids
#         qid_corr2 = [q_ids_problem_solving[key][0] for key in target_q_ids_from_keys]
#
#         assert qid_corr==qid_corr2
#         assert img_id_corr == img_id_corr2
#
#         ques_str_corr = [q_ids_problem_solving[key][2] for key in target_q_ids_from_keys]
#         all_answers_corr = [q_ids_problem_solving[key][3] for key in target_q_ids_from_keys]
#         ques_type_data_corr = [q_ids_problem_solving[key][4] for key in target_q_ids_from_keys]
#         ans_type_data_corr = [q_ids_problem_solving[key][5] for key in target_q_ids_from_keys]
#
#         print(time.time() - st)
#         return qid_corr, pred_ans_corr, softmax_vector_corr, img_id_corr, ques_str_corr, all_answers_corr, ques_type_data_corr, ans_type_data_corr     # ,gt_ans_used



def just_read_standard_files(standard_q_json, standard_a_json):
    st = time.time()

    if str(standard_q_json)[-4:] == 'json':
        with open(standard_q_json) as file:
            details_ques = json.load(file)['questions']
    else:
        with open(standard_q_json, 'rb') as file:
            details_ques = pickle.load(file)['questions']
    q_ids = [details_q['question_id'] for details_q in details_ques]
    img_ids = [details_q['image_id'] for details_q in details_ques]
    ques_str = [details_q['question'] for details_q in details_ques]

    with open(standard_a_json) as file:
        details_ann = json.load(file)['annotations']
    all_answers = [[ans['answer'] for ans in details_ann[i]['answers']] for i in range(len(img_ids))]
    # most_freq_ans = [details_ann[i]['multiple_choice_answer'] for i in range(len(img_ids))]

    q_ids_ann = [details_ann[i]['question_id'] for i in range(len(img_ids))]
    assert q_ids == q_ids_ann
    img_ids_ann = [details_ann[i]['image_id'] for i in range(len(img_ids))]
    assert img_ids == img_ids_ann

    ques_type_data = [details_ann[i]['question_type'] for i in range(len(img_ids))]
    ans_type_data = [details_ann[i]['answer_type'] for i in range(len(img_ids))]
    print(time.time() - st)
    return q_ids,  img_ids, ques_str, all_answers, ques_type_data, ans_type_data



def my_read_old_area(results_old_pkl, standard_q_json, standard_a_json=None):
    st = time.time()
    with open(results_old_pkl, 'rb') as file:
        res_val = pickle.load(file)
    if str(standard_q_json)[-4:] == 'json':
        raise ValueError('to get the areas of objects in the images, please pass the pickle file- json doesnt have the areas')
        #with open(standard_q_json) as file:
        #    details_ques = json.load(file)['questions']
    else:
        with open(standard_q_json, 'rb') as file:
            details_ques = pickle.load(file)['questions']


    ## from results file- model specific
    pred_ans = [details['ans_id'] for details in res_val]
    softmax_vector = [details['ss_vc'] for details in res_val]
    img_id_res = [details['img_id'] for details in res_val]
    qid_res = [details['ques_id'] for details in res_val]
    # gt_ans_used = [details['gt_ans_id_used'] for details in res_val] ## in case of snmn
    # q_ids = [details['ques_id'] for details in res_val]     ### in cases of snmn
    # assert (q_ids == [details_q['question_id'] for details_q in details_ques])

    q_ids = [details_q['question_id'] for details_q in details_ques]
    img_ids = [details_q['image_id'] for details_q in details_ques]
    ques_str = [details_q['question'] for details_q in details_ques]
    area_overlap = [details_q['area_overlap'] for details_q in details_ques]
    area_max_instance = [details_q['area_max_instance'] for details_q in details_ques]
    area_total = [details_q['area_total'] for details_q in details_ques]
    #ipdb.set_trace()
    ### make sure order of qid, img_id consistent between results and standard annotations files
    ## if not, sort them to what is there in the standar files
    if qid_res != q_ids and img_id_res != img_ids:
        std_q_img_id = {}
        for qid_idx, qid in enumerate(q_ids):
            key = str(qid) + str(img_ids[qid_idx])
            std_q_img_id[key] = qid_idx

        res_q_img_id_pans_ss_vc = {}
        for qid_idx, qid in enumerate(qid_res):
            # if 'torch' in img_id_res[qid_idx].type():
            #     img_id_res[qid_idx] = img_id_res[qid_idx].item()
            # if len(str(img_id_res[qid_idx])) < 20:  ## SAAA fix; for edit_set- img_id is string and not torch object
            #     img_id_res[qid_idx] = img_id_res[qid_idx].item()
            key_new = str(qid) + str(img_id_res[qid_idx])
            res_q_img_id_pans_ss_vc[key_new] = pred_ans[qid_idx], softmax_vector[qid_idx], qid, img_id_res[qid_idx]

        pred_ans_corr = [res_q_img_id_pans_ss_vc[key][0] for key in std_q_img_id.keys()]
        softmax_vector_corr = [res_q_img_id_pans_ss_vc[key][1] for key in std_q_img_id.keys()]
        qid_corr = [res_q_img_id_pans_ss_vc[key][2] for key in std_q_img_id.keys()]
        img_id_corr = [res_q_img_id_pans_ss_vc[key][3] for key in std_q_img_id.keys()]
        assert img_id_corr == img_ids
        assert qid_corr == q_ids

        pred_ans = pred_ans_corr
        softmax_vector = softmax_vector_corr

    if standard_a_json is not None:
        with open(standard_a_json) as file:
            details_ann = json.load(file)['annotations']
        all_answers = [[ans['answer'] for ans in details_ann[i]['answers']] for i in range(len(img_ids))]
        # most_freq_ans = [details_ann[i]['multiple_choice_answer'] for i in range(len(img_ids))]
        ques_type_data = [details_ann[i]['question_type'] for i in range(len(img_ids))]
        ans_type_data = [details_ann[i]['answer_type'] for i in range(len(img_ids))]
        print(time.time() - st)
        return q_ids, pred_ans, softmax_vector, img_ids, ques_str, all_answers, ques_type_data, ans_type_data,  area_overlap, area_max_instance, area_total       # ,gt_ans_used
    else:
        print(time.time() - st)
        return q_ids, pred_ans, softmax_vector, img_ids, ques_str, [], [], []  # ,gt_ans_used


def my_read_short(results_old_pkl, standard_q_json):
    st = time.time()
    with open(results_old_pkl, 'rb') as file:
        res_val = pickle.load(file)
    if str(standard_q_json)[-4:] == 'json':
        with open(standard_q_json) as file:
            details_ques = json.load(file)['questions']
    else:
        with open(standard_q_json, 'rb') as file:
            details_ques = pickle.load(file)['questions']

    pred_ans = [details['ans_id'] for details in res_val]
    softmax_vector = [details['ss_vc'] for details in res_val]
    img_id_res = [details['img_id'] for details in res_val]
    qid_res = [details['ques_id'] for details in res_val]

    q_ids = [details_q['question_id'] for details_q in details_ques]
    img_ids = [details_q['image_id'] for details_q in details_ques]

    #ipdb.set_trace()
    ### make sure order of qid, img_id consistent between results and standard annotations files
    ## if not, sort them to what is there in the standar files
    if qid_res != q_ids and img_id_res != img_ids:
        std_q_img_id = {}
        for qid_idx, qid in enumerate(q_ids):
            key = str(qid) + str(img_ids[qid_idx])
            std_q_img_id[key] = qid_idx

        res_q_img_id_pans_ss_vc = {}
        for qid_idx, qid in enumerate(qid_res):
            key_new = str(qid) + str(img_id_res[qid_idx])
            res_q_img_id_pans_ss_vc[key_new] = pred_ans[qid_idx], softmax_vector[qid_idx], qid, img_id_res[qid_idx]

        pred_ans_corr = [res_q_img_id_pans_ss_vc[key][0] for key in std_q_img_id.keys()]
        softmax_vector_corr = [res_q_img_id_pans_ss_vc[key][1] for key in std_q_img_id.keys()]
        qid_corr = [res_q_img_id_pans_ss_vc[key][2] for key in std_q_img_id.keys()]
        img_id_corr = [res_q_img_id_pans_ss_vc[key][3] for key in std_q_img_id.keys()]
        assert img_id_corr == img_ids
        assert qid_corr == q_ids

        pred_ans = pred_ans_corr
        softmax_vector = softmax_vector_corr

    print(time.time() - st)
    return q_ids, pred_ans, softmax_vector, img_ids


def intersect(lst1, lst2):
    if lst1:   ## if not empty
        return(list(set(lst1).intersection(set(lst2))))
    else:
        return lst2


def get_indices_diff_list_suffix(coco_dict_inv, img_ids_edit_val_list, ques_type_edit_val_list, ans_type_edit_val_list, L_norm_diff_list ,\
                                 diff_softmax_for_val_pred_list, CLASS = None, QUES_TYPE=None, ANS_TYPE=None, preselected_indices = None, preselected_string = None):
    chosen_indices_id = []
    chosen_indices_q  = []
    chosen_indices_a  = []
    common_indices = []
    plt_title_suffix = ''

    if ANS_TYPE is not None:
        chosen_indices_a = [i for i, s in enumerate(ans_type_edit_val_list) if s == ANS_TYPE]
        #print('number of IQA triplets related to answer type:', ANS_TYPE , 'is', len(chosen_indices_a))
        common_indices = intersect(common_indices, chosen_indices_a)

        plt_title_suffix = plt_title_suffix + '_' + str(ANS_TYPE)

    if CLASS is not None :
        CLASS_ID = coco_dict_inv[CLASS]
        chosen_indices_id = [i for i, i_id in enumerate(img_ids_edit_val_list) if int(i_id)%100 == CLASS_ID]
        common_indices = intersect(common_indices, chosen_indices_id)
        #print('number of IQA triplets related to removed object:', CLASS, 'is', len(chosen_indices_id))
        plt_title_suffix = plt_title_suffix + '_' + str(CLASS)


    if QUES_TYPE is not None:
        chosen_indices_q = [i for i, s in enumerate(ques_type_edit_val_list) if s == QUES_TYPE]
        #print('number of IQA triplets related to question_type:', QUES_TYPE , 'is', len(chosen_indices_q))
        common_indices = intersect(common_indices, chosen_indices_q)
        # common_indices = intersect(chosen_indices_id, chosen_indices_q)
        # assert common_indices==common_indices2
        plt_title_suffix= plt_title_suffix + '_' + str(QUES_TYPE)

    if preselected_string is not None:
        plt_title_suffix = preselected_string + '_' + plt_title_suffix


    if preselected_indices is not None:
        common_indices = intersect(preselected_indices, common_indices)
        img_ind = [img_ids_edit_val_list[i] for i in common_indices]


    diff_list_norm = [L_norm_diff_list[i] for i in common_indices]
    diff_list_value = [diff_softmax_for_val_pred_list[i] for i in common_indices]
    #print('number of IQA triplets in common:', len(common_indices))
    plt_title_suffix = plt_title_suffix + '_'

    return(common_indices,diff_list_norm, diff_list_value, plt_title_suffix)




def round_percent(item):
    return(round((item*100),3))


def round_list(lst,n):
    return([round(i,n)] for i in lst)


def vqa_score_list(all_answers, pred_ans, ans_vocab_list):
    matching_ans = []
    ans_score_list = []
    for idx in range(len(pred_ans)):
        matching_ans.append([item for item in all_answers[idx] if item == ans_vocab_list[pred_ans[idx]]])
        ans_score_list.append(min(1, float(len(matching_ans[idx]))/3))    ### acc = min(1, float(len(matching_ans))/3)
    return ans_score_list


# def worst_case_acc_old(chosen_qid_list, qid_predans_edit_val, ans_vocab_list, qid_gt_ans_label):  ## all_qid_list = qid_predans_edit_val.keys()
#     worst_case_acc = {}
#     all_case_acc = {}
#
#     ## worst case accuracy
#     worst_acc_list = []
#     best_acc_list = []
#     for ques in list(set(chosen_qid_list)):   #qid_predans_edit_val.keys():
#         ans_edit = qid_predans_edit_val[ques]
#         ans_edit_label = [ans_vocab_list[i] for i in ans_edit]
#         true_ans_label = qid_gt_ans_label[ques]
#         acc = [int(ans in true_ans_label) for ans in ans_edit_label]
#         all_case_acc[ques] = acc
#         worst_case_acc[ques] = min(acc)
#         worst_acc_list.append(min(acc))
#         best_acc_list.append(max(acc))
#     #print('worst case abs', np.sum(worst_acc_list))
#     print("worst case accuracy is ", round_percent(np.sum(worst_acc_list)/len(worst_acc_list)))
#     print("best case accuracy is ", round_percent(np.sum(best_acc_list)/len(worst_acc_list)))
#     return round_percent(np.sum(worst_acc_list)/len(worst_acc_list)),  round_percent(np.sum(best_acc_list)/len(worst_acc_list))


def worst_case_acc(chosen_qid_list, qid_predans_idx_edit_val, qid_predans_edit_val, ans_vocab_list, qid_gt_ans_label, if_print = None):  ## all_qid_list = qid_predans_edit_val.keys()

    worst_case_idx = {}
    best_case_idx = {}
    # worst_case_indices = {}
    # best_case_indices = {}
    off_worst_case_idx = {}
    off_best_case_idx = {}
    # off_worst_case_indices = {}
    # off_best_case_indices = {}

    #     qid_predans_idx_val = {}
    #     for idx, a in enumerate(qid_val):
    #         qid_predans_idx_val.setdefault(a, []).append(idx)

    ## worst case accuracy
    worst_acc_list = []
    best_acc_list = []
    off_worst_acc_list = []
    off_best_acc_list = []

    for ques in list(set(chosen_qid_list)):  # qid_predans_edit_val.keys():

        ans_edit = qid_predans_edit_val[ques]
        ans_edit_label = [ans_vocab_list[i] for i in ans_edit]
        true_ans_label = qid_gt_ans_label[ques]

        corresponding_idx_list = qid_predans_idx_edit_val[ques]

        acc = [int(ans in true_ans_label) for ans in ans_edit_label]
        assert len(acc) == len(corresponding_idx_list)

        matching_ans = []
        off_acc = []
        for idx, pred_ans in enumerate(ans_edit_label):
            matching_ans.append([item for item in true_ans_label if item == pred_ans])
            off_acc.append(min(1, float(len(matching_ans[idx])) / 3))



            # all_ind_where_acc_0 = [acc.index(i) for i in acc if i==min(acc)] #or [i for i in range(len(acc)) if acc[i]==0]
        # all_ind_where_acc_1 = [acc.index(i) for i in acc if i==max(acc)]
        one_worst_idx = acc.index(min(acc))  ## by default takes the first one
        one_best_idx = acc.index(max(acc))

        final_worst_idx = corresponding_idx_list[one_worst_idx]
        final_best_idx = corresponding_idx_list[one_best_idx]

        # off_all_ind_where_acc_0 = [off_acc.index(i) for i in off_acc if i==min(off_acc)] #or [i for i in range(len(acc)) if acc[i]==0]
        # off_all_ind_where_acc_1 = [off_acc.index(i) for i in off_acc if i==max(off_acc)]
        off_one_worst_idx = off_acc.index(min(off_acc))  ## by default takes the first one
        off_one_best_idx = off_acc.index(max(off_acc))

        #if min(off_acc)!= max(off_acc):
        #    ipdb.set_trace()
        #    print(ques)
        off_final_worst_idx = corresponding_idx_list[off_one_worst_idx]
        off_final_best_idx = corresponding_idx_list[off_one_best_idx]

        # dictionaries
        worst_case_idx[ques] = final_worst_idx  # worst_index
        best_case_idx[ques] = final_best_idx  # best_index
        off_worst_case_idx[ques] = off_final_worst_idx  # worst_index
        off_best_case_idx[ques] = off_final_best_idx  # best_index

        # worst_indices = [corresponding_idx_list[i] for i in all_ind_where_acc_0]
        # best_indices =  [corresponding_idx_list[i] for i in all_ind_where_acc_1]
        # worst_case_indices[ques] = worst_indices
        # best_case_indices[ques] = best_indices

        # lists
        worst_acc_list.append(min(acc)),
        best_acc_list.append(max(acc))
        off_worst_acc_list.append(min(off_acc)),
        off_best_acc_list.append(max(off_acc))

    # print('worst case abs', np.sum(worst_acc_list))
    if if_print:
        print("worst case accuracy ({}): ".format(len(worst_acc_list)), round_percent(np.sum(worst_acc_list) / len(worst_acc_list)))
        print("best case accuracy ({}): ".format(len(best_acc_list)), round_percent(np.sum(best_acc_list) / len(best_acc_list)))
        print("worst case official accuracy ({}): ".format(len(off_worst_acc_list)), round_percent(np.sum(off_worst_acc_list) / len(off_worst_acc_list)))
        print("best case official accuracy ({}): ".format(len(off_best_acc_list)), round_percent(np.sum(off_best_acc_list) / len(off_best_acc_list)))

    return round_percent(np.sum(worst_acc_list) / len(worst_acc_list)), \
           round_percent(np.sum(best_acc_list) / len(worst_acc_list)), \
           worst_case_idx, best_case_idx, off_worst_case_idx, off_best_case_idx



def vqa_score_list(all_answers, pred_ans, ans_vocab_list):
    matching_ans = []
    ans_score_list = []
    for idx in range(len(pred_ans)):
        matching_ans.append([item for item in all_answers[idx] if item == ans_vocab_list[pred_ans[idx]]])
        ans_score_list.append(min(1, float(len(matching_ans[idx]))/3))    ### acc = min(1, float(len(matching_ans))/3)
    return ans_score_list


# wrst_lst = [worst_case_acc[key] for key in worst_case_acc.keys()]
# print('worst case abs',np.sum(wrst_lst))
# print('worst case percent',round_percent(np.sum(wrst_lst)/len(wrst_lst)))

def ch_atleast_once(chosen_qid_list, qid_predans_edit_val, qid_predans_val ,ans_vocab_list):
    atleast_one_ans_ch = np.zeros(len(set(chosen_qid_list)))
    for idx, ques in enumerate(list(set(chosen_qid_list))):
        ans_edit_val = set(qid_predans_edit_val[ques])
        ans_val = set(qid_predans_val[ques])      ## not a set , every question unique in val- atleast has unique q_id
        ans_val_label = [ans_vocab_list[i] for i in list(ans_val)]
        ans_edit_val_labels = [ans_vocab_list[i] for i in list(ans_edit_val)]
        #if len(list(ans_edit_val-ans_val))>0:
        #    atleast_one_ans_ch[idx] += 1
        if ans_edit_val_labels!= ans_val_label:
            atleast_one_ans_ch[idx] +=1
    print('Of',len(atleast_one_ans_ch), 'unique questions in edited set,', np.sum(atleast_one_ans_ch),
          'change answer_labels atleast once' ,'   ', round_percent(np.sum(atleast_one_ans_ch)/len(atleast_one_ans_ch)), '%')
    return (atleast_one_ans_ch)

#    chosen_qid_list = [qid_edit_val[i] for i in chosen_indices]
#     chosen_atleast_one_ans_ch =np.zeros(len(set(chosen_qid_list)))
#     for idx, ques in enumerate(list(set(chosen_qid_list))):
#         ans_edit_val = set(qid_predans_edit_val[ques])
#         ans_val = set(qid_predans_val[ques])      ## not a set , every question unique in val- atleast has unique q_id
#         ans_val_label = [ans_vocab_list[i] for i in list(ans_val)]
#         ans_edit_val_labels = [ans_vocab_list[i] for i in list(ans_edit_val)]
#         #if len(list(ans_edit_val-ans_val))>0:
#         #    atleast_one_ans_ch[idx] += 1
#         if ans_edit_val_labels!= ans_val_label:
#             chosen_atleast_one_ans_ch[idx] +=1


