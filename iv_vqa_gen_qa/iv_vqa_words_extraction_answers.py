import json
import time
import config
import nltk
from vocab_mapping import nouns_matched_COCO, nouns_matched_COCO_stuff
from nltk.stem.wordnet import WordNetLemmatizer

lem = WordNetLemmatizer()

with open('coco-labels/coco_labels_80_classes.txt') as f:
    # return the split results, which is all the words in the file.
    coco_80 = f.read().splitlines()  ###coco 80 classes

# with open('all_coco.txt') as f:
#     # return the split results, which is all the words in the file.
#     coco_stuff_91= f.read().splitlines()      ##coco_stuff 91 classes


def ans_list(answer_file):
    start = time.time();
    with open(answer_file) as f:
        annotations = json.load(f)["annotations"]
    ans_n = []
    for ann in annotations:
        #print(ann["image_id"])
        #print(ann["question_id"])
        all_ans = []
        for answer in ann["answers"]:
            words = nltk.word_tokenize(answer["answer"])
            words = [word.lower() for word in words]
            words = [lem.lemmatize(w) for w in words]
            all_ans.append(words)
        ans_n.append([all_ans[i][0] for i in range(0,10)])
    print(time.time() - start)
    return(ans_n)



ann_coco_file_val = config.coco_ann_dir + 'instances_val2014.json'
ann_coco_file_train = config.coco_ann_dir + 'instances_train2014.json'
with open(ann_coco_file_train) as f:
    ann_coco_1 = json.load(f)['categories']


id_list = []
class_list = []
for cate in ann_coco_1:
    id_list.append(cate['id'])
    class_list.append(cate['name'])


def ready_json(answer_file, filename, nouns_list):
    start = time.time();
    a1 = nouns_matched_COCO(nouns_list, coco_80, id_list, class_list)  ##here attach class_id for person, if man/woman was there
    # a3 = nouns_matched_COCO_stuff(nouns_list, coco_stuff_91)
    abcd = []
    file_data = {}

    with open(answer_file) as f:
        annotations = json.load(f)["annotations"]  ##read json
    for n, ann in enumerate(annotations):
        #print(n)
        abcd.append({"question_type": ann['question_type'],
                     "multipe_choice_answer": ann['multiple_choice_answer'],
                     "answers": ann['answers'],  ##answer_list[n] if only answers you want to save
                     "image_id": ann['image_id'],
                     "answer_type": ann['answer_type'],
                     "question_id": ann['question_id'],
                     "ans_match_COCO": a1[n]})
        # "ans_match_COCO_stuff": a3[n]})
    file_data['answers'] = abcd
    with open(filename, 'w') as outfile_val1:
        json.dump(file_data, outfile_val1)
    print(time.time() - start)


answer_file_1 = config.vqa_a_dir + "v2_mscoco_train2014_annotations.json"
nouns_list_1 = ans_list(answer_file_1)
ready_json(answer_file_1,'tagged_train2014_answers.json', nouns_list_1)


answer_file_2 = config.vqa_a_dir + "v2_mscoco_val2014_annotations.json"
nouns_list_2 = ans_list(answer_file_2)
ready_json(answer_file_2, 'tagged_val2014_answers.json', nouns_list_2)




