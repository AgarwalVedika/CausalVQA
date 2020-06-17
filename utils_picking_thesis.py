import json
import pickle
import time
import random
random.seed(1234)
import numpy as np; np.random.seed(1234)
import ipdb


def my_read_old(results_old_pkl, standard_q_json, standard_a_json=None):
    st = time.time()
    
    #ipdb.set_trace()
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
    img_ids = [str(iid).zfill(12) for iid in img_ids]
    ques_str = [details_q['question'] for details_q in details_ques]

    if  isinstance(img_id_res[0], int):   ### FIX for old files- as snmn then is the best... 1e-3 lr...
        img_id_res = [str(i).zfill(12) for i in img_id_res]

    ### make sure order of qid, img_id consistent between results and standard annotations files
    ## if not, sort them to what is there in the standard files
    #ipdb.set_trace()
    if qid_res != q_ids and img_id_res != img_ids:
        std_q_img_id = {}

        ### SAA and CNNN_LSTM in case for train set: we find a discrpancy so qids are nto a strict subset of qid_res
        if len((set(q_ids) & set(qid_res))) != len(set(q_ids)):
            print()
            print(' DISCREPANCY ALERT ')
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

        #ipdb.set_trace()
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
