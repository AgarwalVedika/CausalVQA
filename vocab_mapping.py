import numpy as np
import json
import re
from collections import defaultdict
import matplotlib.pyplot as plt
import time
import nltk
import ipdb
from nltk import pos_tag

from nltk.stem.wordnet import WordNetLemmatizer
lem = WordNetLemmatizer()

from nltk.probability import FreqDist
#
# from skimage.morphology import disk, square
# import scipy
# selem = disk(2)

#
# def plot_comparison(original, filtered, filtered_5 , filter_name, filter_name_5):
#
#     fig, (ax1, ax2, ax3) = plt.subplots(ncols=3, figsize=(8, 4), sharex=True,
#                                    sharey=True)
#     ax1.imshow(original, cmap=plt.cm.gray)
#     ax1.set_title('original')
#     ax1.axis('off')
#     ax2.imshow(filtered, cmap=plt.cm.gray)
#     ax2.set_title(filter_name)
#     ax2.axis('off')
#     ax3.imshow(filtered_5, cmap=plt.cm.gray)
#     ax3.set_title(filter_name_5)
#     ax3.axis('off')
# #plot_comparison(maskTotal_per_cat_id_all_ins[67], maskTotal_per_cat_id_all_ins_dilated[67],
# #                 maskTotal_per_cat_id_all_ins_dilated_5[67], 'dilation', 'dilation_5')



# def overlap_criterion(obj_q, cls_img_set, maskTotal_per_cat_id_all_ins_dil):
#     # OVERLAP CRITERION
#     ### if object referred in question overlaps with any object in the image=
#     ## dont discard that other object!!
#     masks_intersect = {}
#     obj_not_rem_overlap_metric = []
#     if len(obj_q)!= 0 :
#         for cls_id in obj_q:
#             if cls_id in maskTotal_per_cat_id_all_ins_dil.keys():   ## might happen the coc segmentation doesnt exist of the object referred in ques/ans: VQA and COCO ann intersection
#                 mask_q = maskTotal_per_cat_id_all_ins_dil[cls_id]
#                 cls_img_set.remove(cls_id)
#                 for cat_id in cls_img_set:
#                     mask_other = maskTotal_per_cat_id_all_ins_dil[cat_id]
#                     score = round(np.sum(mask_q * mask_other) / (np.sum(mask_q) + np.sum(mask_other)),3)
#                     if score:
#                         masks_intersect[cls_id, cat_id] = score
#                         #obj_not_rem_overlap_metric.append(cat_id)
#                 obj_q.remove(cls_id)
#             else:
#                 print('COCO and VQA annotations dont intersect')
#         return masks_intersect
#     else:
#         return None


def nouns_matched_COCO_stuff(nouns_list, coco_stuff_91):  ### yet to map it to ids
    a3 = []
    for line in nouns_list:
        a2 = []
        for word in line:
            if word in coco_stuff_91:
                a2.append(word)
        a3.append(a2)
    print(len(a3))
    return (a3)


def get_nouns_old(question_file):
    start = time.time();
    #SENTENCE_SPLIT_REGEX = re.compile(r'(\W+)')
    all_nouns_list = []

    with open(question_file) as f:
        questions = json.load(f)['questions']                   ##read json
    for n_q, q in enumerate(questions):
        #words = SENTENCE_SPLIT_REGEX.split(q['question'].lower())
        #words = [w.strip() for w in words if len(w.strip()) > 0]
        words = nltk.word_tokenize(q['question'])
        nouns = [word for (word, tag) in pos_tag(words) if (tag == 'NN' or tag == 'NNS' or tag == 'PRP') ]   ###getting_nouns & pronouns
        nouns = [lem.lemmatize(w) for w in nouns]        ###lemmatizing nouns after- as they are fewer words
        all_nouns_list.append(nouns)
    print(time.time() - start)
    return(all_nouns_list)


def get_words(question_file):
    start = time.time();
    #SENTENCE_SPLIT_REGEX = re.compile(r'(\W+)')
    #all_nouns_list = []
    all_words_list = []
    with open(question_file) as f:
        questions = json.load(f)['questions']                   ##read json
    for n_q, q in enumerate(questions):
        words = nltk.word_tokenize(q['question'])
        words = [word.lower() for word in words]
        words = [lem.lemmatize(w) for w in words]        ###lemmatizing  plural to singular form
        all_words_list.append(words)
    print(time.time() - start)
    return(all_words_list)


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


### so now check if these tagged nouns are in the coco_classes: if yes, then append
## don't deal with words- deal with class ids- better way
def nouns_matched_COCO(nouns_list, coco_80, id_lst, class_list):  ########maps common classes to ids / not classes here
    a1 = []
    for line in nouns_list:
        a2 = []
        for i, word in enumerate(line):
            if ("man" in word or "woman" in word or "player" in word or "child" in word or "girl" in word or  ### many mapped to word too then -_-
                    word == 'boy' or word == 'people' or word == 'lady' or word == 'guy' or word == 'kid' or
                    word == 'he' or word == 'she' or word == 'I' or word == 'surfer' or word == 'his' or word == 'her' or
                    word == 'cop' or word == 'soldier' or word =='police' or 'catcher' in word or 'pitcher' in word or
                    word == 'baby' or word == 'men' or word == 'couple' or word == "biker" or word == "shirt" or
                    word == "jacket" or word == "dress" or word == "spectator" or word == "rider" or word == "batter" or
                    word == "they" or word == "outfit" or word == "helmet" or word=="anyone" or word=="someone" or word=="wetsuit" or word=="gay" or
                    word == "reporter" or word=="somebody" or word=="anybody" or word=="everyone" or word=="pants" or word =="coats" or
                    word=="jackets" or "wristband" in word or "team" in word or word=="wearing" or word=="clothes" or word=="sneakers" or word=="dancing"):
                a2.append(id_lst[class_list.index('person')])  ##a2.append('person')  id_lst[class_list.index(word)]

            if ("vehicle" in word):
                a2.append(id_lst[class_list.index('bicycle')])
                a2.append(id_lst[class_list.index('car')])
                a2.append(id_lst[class_list.index('motorcycle')])
                a2.append(id_lst[class_list.index('airplane')])
                a2.append(id_lst[class_list.index('bus')])
                a2.append(id_lst[class_list.index('train')])
                a2.append(id_lst[class_list.index('truck')])
                a2.append(id_lst[class_list.index('boat')])

            if ("plane" in word or word == 'jet' or word == 'tail' or 'airline' in word or word == 'aircraft' or word == "hangar"
                    or word == 'airport' or word == 'military' or word == 'commercial' or word == 'runway' or "wing" in word):
                a2.append(id_lst[class_list.index('airplane')])
            if ("bike" in word or "biking" in word or "cycling" in word):
                a2.append(id_lst[class_list.index('bicycle')])
            if ("bike" in word or "motor" in word):
                a2.append(id_lst[class_list.index('motorcycle')])
            if ("van" in word or word=="trolley"):
                a2.append(id_lst[class_list.index('bus')])
            if ("van" in word or "taxi" in word or "trunk" in word or word=="truck" or word=="suv" or word=="SUV"):  ### car and truck -_-
                a2.append(id_lst[class_list.index('car')])
            if (word == 'engine' or "tram" in word or word == 'trolley' or word == 'subway'):
                a2.append(id_lst[class_list.index('train')])

            #             if ("traffic" in word and "light" in line[i+1]):           ## dealing with composite words
            #                 a2.append(id_lst[class_list.index('traffic light')])
            if ("light" in word or "traffic" in word):
                a2.append(id_lst[class_list.index('traffic light')])

            #             if ("stop" in word and "sign" in line[i+1]):           ## dealing with composite words
            #                 a2.append(id_lst[class_list.index('stop sign')])
            if ("sign" in word or "stop" in word):
                a2.append(id_lst[class_list.index('stop sign')])

            if ("meter" in word):
                a2.append(id_lst[class_list.index('parking meter')])

            if ("hydrant" in word or 'hydrate' in word or 'hydra' in word):
                a2.append(id_lst[class_list.index('fire hydrant')])
            #             if ("fire" in word and "hydrant" in line[i+1]):           ## dealing with composite words
            #                 a2.append(id_lst[class_list.index('fire hydrant')])

            if ("animal" in word or "baby" in word or "cub" in word or "cubs" in word or "babies" in word or
                    "paw" in word or "zoo" in word or "circus" in word or word == "they" or word == "Africa" or "leg" in word
                    or "creature" in word or word == "africa" or word=="herd"):
                a2.append(id_lst[class_list.index('bird')])
                a2.append(id_lst[class_list.index('cat')])
                a2.append(id_lst[class_list.index('dog')])
                a2.append(id_lst[class_list.index('horse')])
                a2.append(id_lst[class_list.index('sheep')])
                a2.append(id_lst[class_list.index('cow')])
                a2.append(id_lst[class_list.index('elephant')])
                a2.append(id_lst[class_list.index('bear')])
                a2.append(id_lst[class_list.index('zebra')])
                a2.append(id_lst[class_list.index('giraffe')])
            if ("beak" in word or "duck" in word or "goose" in word or "gull" in word or "pigeon" in word or "chicken" in word or "penguin" in word or word=="song"):
                a2.append(id_lst[class_list.index('bird')])
            if ("kitty" in word or "kitten" in word):
                a2.append(id_lst[class_list.index('cat')])
            if ("puppy" in word or word=="puppies"):
                a2.append(id_lst[class_list.index('dog')])
            if ("lamb" in word):
                a2.append(id_lst[class_list.index('sheep')])
            if ("calf" in word):
                a2.append(id_lst[class_list.index('elephant')])
                a2.append(id_lst[class_list.index('giraffe')])
            if ("pet" in word or "collar" in word):
                a2.append(id_lst[class_list.index('dog')])
            if ("horse" in word or "riding" in word or word=="pony" or "foal" in word):
                a2.append(id_lst[class_list.index('horse')])
            if ("stripe" in word or "Zebra" in word):
                a2.append(id_lst[class_list.index('zebra')])
            if ("Giraffe" in word):
                a2.append(id_lst[class_list.index('giraffe')])
            if ("cattle" in word or word=="oxen" or word=="ox" or word=="herd" or word=="calves" or "bull" in word or "calf" in word):
                a2.append(id_lst[class_list.index('cow')])
            if ("Bear" in word):
                a2.append(id_lst[class_list.index('bear')])



            if ("rain" in word):
                a2.append(id_lst[class_list.index('umbrella')])
            if ("bag" in word):
                a2.append(id_lst[class_list.index('handbag')])
                a2.append(id_lst[class_list.index('suitcase')])




            if ("disc" in word or "frisbee" in word or word=="disk"):
                a2.append(id_lst[class_list.index('frisbee')])

            #             if ("sports" in word and "ball" in line[i+1]):           ## dealing with composite words
            #                 a2.append(id_lst[class_list.index('sports ball')])
            #             if ("baseball" in word and "bat" in line[i+1]):           ## dealing with composite words
            #                 a2.append(id_lst[class_list.index('baseball bat')])
            #             if ("baseball" in word and "glove" in line[i+1]):           ## dealing with composite words
            #                 a2.append(id_lst[class_list.index('baseball glove')])

            if ("ball" in word):
                a2.append(id_lst[class_list.index('sports ball')])
            if (word == "bat"):
                a2.append(id_lst[class_list.index('baseball bat')])
            if ("glove" in word):
                a2.append(id_lst[class_list.index('baseball glove')])
            if ("skate" in word):
                a2.append(id_lst[class_list.index('skateboard')])
            if ("surf" in word):
                a2.append(id_lst[class_list.index('surfboard')])
            if ("board" in word):
                a2.append(id_lst[class_list.index('skateboard')])
                a2.append(id_lst[class_list.index('surfboard')])
                a2.append(id_lst[class_list.index('snowboard')])
            if ("ski" in word):
                a2.append(id_lst[class_list.index('skis')])
            if ("baseball" in word):
                a2.append(id_lst[class_list.index('baseball bat')])
                a2.append(id_lst[class_list.index('baseball glove')])
            if ("racket" in word or "tennis" in word or word == "racquet"):
                a2.append(id_lst[class_list.index('tennis racket')])
            if ("kite" in word or "fly" in word or word=="sky"):   ## flying kite fairly common answer
                a2.append(id_lst[class_list.index('kite')])


                #ipdb.set_trace()
            #             if ("wine" in word and "glass" in line[i+1]):           ## dealing with composite words
            #                 a2.append(id_lst[class_list.index('wine glass')])
            #             if ("glass" in word and "wine" not in line[i-1]):        ### avoiding multiple occurences for same word
            #                 a2.append(id_lst[class_list.index('wine glass')])
            if ("wine" in word or "glass" in word or word=="bottle" or "beverage" in word or word=="drink"):
                a2.append(id_lst[class_list.index('wine glass')])
            if ("wine" in word or word=="bottle" or "thermos" in word or "flask" in word or word=="drink" or word=="beer" or "beverage" in word):
                a2.append(id_lst[class_list.index('bottle')])
            if ("wine" in word or word=="glass" or word=="drink" or word=="mug" or word=="beverage" or word=="coffee" or word=="tea"):
                a2.append(id_lst[class_list.index('cup')])

            if ("silverware" in word):
                a2.append(id_lst[class_list.index('spoon')])



            if ("fruit" in word or word == "food" or word == "eating"):
                a2.append(id_lst[class_list.index('bowl')])
                a2.append(id_lst[class_list.index('banana')])
                a2.append(id_lst[class_list.index('apple')])
                a2.append(id_lst[class_list.index('hot dog')])
                a2.append(id_lst[class_list.index('pizza')])
                a2.append(id_lst[class_list.index('cake')])
                a2.append(id_lst[class_list.index('sandwich')])
                a2.append(id_lst[class_list.index('orange')])
                a2.append(id_lst[class_list.index('broccoli')])
                a2.append(id_lst[class_list.index('carrot')])
                a2.append(id_lst[class_list.index('donut')])

            # if (word=="yellow" or "nutritious" in word):
            #     a2.append(id_lst[class_list.index('banana')])

            if ("hot dog" in word or "hot" in word or "dog" in word):
                a2.append(id_lst[class_list.index('hot dog')])

            #             if (word == 'hot' and "dog" in line[i+1]):           ## dealing with composite words
            #                 a2.append(id_lst[class_list.index('hot dog')])
            if ("doughnut" in word or "donut" in word or "dough" in word):
                a2.append(id_lst[class_list.index('donut')])
            if ("dessert" in word or "frosting" in word or "cake" in word or "cut" in word):  ## cutting cake fairly common answer
                a2.append(id_lst[class_list.index('cake')])
                #ipdb.set_trace()
            if ("topping" in word or word=="slice"):  ## dealing with composite words
                a2.append(id_lst[class_list.index('pizza')])
            if (word=="ketchup"):  ## dealing with composite words
                a2.append(id_lst[class_list.index('sandwich')])



            if ("furniture" in word):
                a2.append(id_lst[class_list.index('chair')])
                a2.append(id_lst[class_list.index('couch')])
                a2.append(id_lst[class_list.index('potted plant')])
                a2.append(id_lst[class_list.index('bed')])
                a2.append(id_lst[class_list.index('dining table')])
                a2.append(id_lst[class_list.index('toilet')])


            if ("table" in word or word=="desk"):
                a2.append(id_lst[class_list.index('dining table')])
            if ("seat" in word):
                a2.append(id_lst[class_list.index('couch')])
                a2.append(id_lst[class_list.index('bench')])
                a2.append(id_lst[class_list.index('chair')])
            if ("stool" in word):
                a2.append(id_lst[class_list.index('chair')])

            if ("pot" in word or "plant" in word or "flower" in word or "vase" in word):
                a2.append(id_lst[class_list.index('potted plant')])
                a2.append(id_lst[class_list.index('vase')])


            if ("electronic" in word):
                a2.append(id_lst[class_list.index('tv')])
                a2.append(id_lst[class_list.index('laptop')])
                a2.append(id_lst[class_list.index('mouse')])
                a2.append(id_lst[class_list.index('remote')])
                a2.append(id_lst[class_list.index('keyboard')])
                a2.append(id_lst[class_list.index('cell phone')])


            if ("television" in word):
                a2.append(id_lst[class_list.index('tv')])
            if ("computer" in word or "monitor" in word):
                a2.append(id_lst[class_list.index('laptop')])
            if ("screen" in word):
                a2.append(id_lst[class_list.index('laptop')])
                a2.append(id_lst[class_list.index('tv')])
                a2.append(id_lst[class_list.index('apple')])

            if ("bed" in word):
                a2.append(id_lst[class_list.index('bed')])

            if ("phone" in word):
                a2.append(id_lst[class_list.index('cell phone')])

            if ("appliance" in word):
                a2.append(id_lst[class_list.index('microwave')])
                a2.append(id_lst[class_list.index('oven')])
                a2.append(id_lst[class_list.index('toaster')])
                a2.append(id_lst[class_list.index('sink')])
                a2.append(id_lst[class_list.index('refrigerator')])


            if ("fridge" in word):
                a2.append(id_lst[class_list.index('refrigerator')])

            if ("book" in word or "novel" in word):
                a2.append(id_lst[class_list.index('book')])

            if ("pot" in word or "vase" in word or "flower" in word):
                a2.append(id_lst[class_list.index('vase')])

            if ("time" in word):
                a2.append(id_lst[class_list.index('clock')])

            if ("scissor" in word or "cut" in word):
                a2.append(id_lst[class_list.index('scissors')])

            if ("brush" in word):
                a2.append(id_lst[class_list.index('toothbrush')])
            if ("kite" in word):
                a2.append(id_lst[class_list.index('kite')])

            if ("drier" in word):
                a2.append(id_lst[class_list.index('hair drier')])
            #             if ("hair" in word and "drier" in line[i+1]):           ## dealing with composite words
            #                 a2.append(id_lst[class_list.index('hair drier')])
            if ("teddy" in word or "toy" in word or "bear" in word or "doll" in word):
                a2.append(id_lst[class_list.index('teddy bear')])

            if word in coco_80:
                a2.append(id_lst[class_list.index(word)])

            for coco_word in coco_80:
                if coco_word in word:
                    a2.append(id_lst[class_list.index(coco_word)])
                    # if coco_word=="cake" or coco_word=="kite":
                    #     ipdb.set_trace()

            #ipdb.set_trace()

        a1.append(a2)
    print(len(a1))
    return (a1)
