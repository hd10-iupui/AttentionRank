import csv
import os
import time
import numpy as np
import pickle


def load_pickle(fname):
    with open(fname, "rb") as f:
        return pickle.load(f, encoding="latin1")  # add, encoding="latin1") if using python3 and downloaded data


def weights_comb(input_weights, strategy=3):
    if strategy == 1:
        comb_weight = np.array(input_weights).mean()
    elif strategy == 2:
        comb_weight = np.array(input_weights).max()
    elif strategy == 3:
        comb_weight = np.array(input_weights).sum()
    else:
        comb_weight = np.array(input_weights).prod() / np.array(input_weights).sum()
    return comb_weight


def data_iterator():
    for i, doc in enumerate(data):
        # if i % 100 == 0 or i == len(data) - 1:
        # print("{:.1f}% done".format(100.0 * (i + 1) / len(data)))
        yield doc["tokens"], np.array(doc["attns"])


def get_data_points(head_data):
    xs, ys, avgs = [], [], []
    for layer in range(12):
        for head in range(12):
            ys.append(head_data[layer, head])
            xs.append(1 + layer)
        avgs.append(head_data[layer].mean())
    return xs, ys, avgs


def is_number(s):
    try:
        float(s)
        return True
    except ValueError:
        pass

    try:
        import unicodedata
        unicodedata.numeric(s)
        return True
    except (TypeError, ValueError):
        pass

    return False


def merge_dict(x, y):
    for k, v in x.items():
        if k in y.keys():
            y[k] = y[k] + v
        else:
            y[k] = v


def map_attn(example, heads, sentence_length, layer_weight, record):
    counter_12 = 0  # check head index

    current_sum_attn = []
    for ei, (layer, head) in enumerate(heads):
        attn = example["attns"][layer][head]  # [0:sentence_length, 0:sentence_length]
        attn = np.array(attn)
        attn /= attn.sum(axis=-1, keepdims=True)  # norm each row
        attn_sum = attn.sum(axis=0, keepdims=True)  # add up 12 heads # np.shape(attn_sum) = (1,sentence length)
        words = example["tokens"]  # [0:sentence_length]
        weights_list = attn_sum[0]

        single_word_dict = {}
        for p in range(len(words)):
            hashtag_lead = words[p]
            hash_weights = [weights_list[p]]
            weight_new = weights_comb(hash_weights, 3) * layer_weight
            single_word_dict[hashtag_lead + "_" + str(p)] = weight_new  # p is the word position in sentence

        current_sum_attn.append(np.array(list(single_word_dict.values())))  # dict.values() keep the entries order
        # shape(current_sum_attn) = (12, words number), each item in it is a list of words attn in current sentence
        counter_12 += 1  # check head index

        if counter_12 % 12 == 0:  # if head number get 12, sum all 12 heads attn and output
            head = np.sum(current_sum_attn, axis=0)  # dict zip can read array, do not need to list it
            # double check
            # print(sum([item[0] for item in current_sum_attn]))
            # print(head[0])
            longer_key_list = []
            current_key_list = list(single_word_dict.keys())  # [words_positions] # dict.keys() keep the entries order
            for each_key in current_key_list:
                longer_key_list.append(each_key + "_" + str(record))  # word_positionInSentence_sentenceNumber
            current_dict = dict(zip(longer_key_list, head))  # head = word_p_s attn, heads are already summed here

            return current_dict


# main
start_time = time.time()

dataset = 'SemEval2017'
text_path = dataset + '/processed_docsutf8/'
output_path = dataset + '/processed_' + dataset + '/'

files = os.listdir(text_path)
for i, file in enumerate(files):
    files[i] = file[:-4]

files = files[:]

save_path = output_path + "token_attn_paired/attn/"
if not os.path.exists(save_path):
    os.makedirs(save_path)

bert_name = "orgbert"

for n, file in enumerate(files):
    attn_extracting_dir = output_path + "sentence_paired_text/" + file + '_' + bert_name + '_attn.pkl'
    data = load_pickle(attn_extracting_dir)

    w = csv.writer(open(save_path + file + "token_attn_paired.csv", "w"))

    for r in range(len(data)):
        record = r
        sentence_length = len(data[record]['tokens'])

        # consider the 12th layer
        weight = 1
        layer = 11
        sentence_dict = map_attn(data[record],
                                 [(layer, 0), (layer, 1), (layer, 2), (layer, 3),
                                   (layer, 4), (layer, 5), (layer, 6), (layer, 7),
                                   (layer, 8), (layer, 9), (layer, 10), (layer, 11)],
                                 sentence_length, weight, record)

        # print('words:', len(document_dict))

        for k, v in sentence_dict.items():
            cut = k.find("_")  # although we do not keep word position, word position helps to keep its original order
            k_short = k[0:cut]
            w.writerow([k_short, v])

    run_time = time.time()
    print(n + 1, "th file", file, "running time", run_time - start_time)
