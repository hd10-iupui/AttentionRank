import csv
import os
import time
import numpy as np
import torch
from torch import nn
import sys
from string import punctuation
from nltk.corpus import stopwords
from tqdm import tqdm


punctuations = []
for punc in range(len(punctuation)):
    punctuations.append(punctuation[punc])
stop_words = stopwords.words('english')
maxInt = sys.maxsize

while True:
    # decrease the maxInt value by factor 10
    # as long as the OverflowError occurs.
    try:
        csv.field_size_limit(maxInt)
        break
    except OverflowError:
        maxInt = int(maxInt/10)


def cosine_similarity(x, y, norm=False):
    assert len(x) == len(y), "len(x) != len(y)"
    res = np.array([[x[i] * y[i], x[i] * x[i], y[i] * y[i]] for i in range(len(x))])
    cos = sum(res[:, 0]) / (np.sqrt(sum(res[:, 1])) * np.sqrt(sum(res[:, 2])))
    return 0.5 * cos + 0.5 if norm else cos  # [0, 1]


def self_attn_matrix(embedding_set):
    ls = np.shape(embedding_set)[0]
    Q = torch.tensor(embedding_set)
    K = torch.tensor(embedding_set)
    V = torch.tensor(embedding_set)
    attn = torch.matmul(Q, K.transpose(-1,-2))
    attn = nn.Softmax(dim=1)(attn)
    V = torch.matmul(attn, V)
    V = torch.sum(V, dim=0)
    V = V/ls
    return V


def cross_attn_matrix(D,Q):
    D = torch.tensor(D)
    Q = torch.tensor(Q)
    attn = torch.matmul(D, Q.transpose(-1,-2))
    S_d2q = nn.Softmax(dim=1)(attn)  # S_d2q : softmax the row; shape[len(doc), len(query)]
    S_q2d = nn.Softmax(dim=0)(attn)  # S_q2d : softmax the col; shape[len(doc), len(query)]
    A_d2q = torch.matmul(S_d2q, Q)
    A_q2d = torch.matmul(S_d2q, torch.matmul(S_q2d.transpose(-1,-2), D))
    V = (D+A_d2q+torch.mul(D, A_d2q)+torch.mul(D, A_q2d))
    V = V/4
    return V


start_time = time.time()

dataset = 'SemEval2017'

text_path = dataset + '/processed_docsutf8/'
output_path = dataset + '/processed_' + dataset+ '/'
save_path = output_path + 'candidate_cross_attn_value/'
if not os.path.exists(save_path):
    os.makedirs(save_path)

# get files name
files = os.listdir(text_path)
for i, file in enumerate(files):
    files[i] = file[:-4]


# run all files
for n, file in enumerate(files):

    # doc embedding set
    embedding_path = output_path + 'doc_word_embedding_by_sentences/' + file + '/'
    sentence_files = os.listdir(embedding_path)  # get sentence list

    all_sentences_word_embedding = []
    for sentence_file in sentence_files:  # do not need to sort
        sentence_word_embedding = []
        with open(embedding_path + sentence_file, newline='') as csvfile:
            spamreader = csv.reader(csvfile, delimiter=',')
            for row in spamreader:
                # print(row[1])
                value_list = row[1][1:-1]
                if value_list[0] == ' ':
                    value_list = value_list[1:]
                if value_list[-1] == ' ':
                    value_list = value_list[:-1]
                value_list = value_list.replace('\n', '').replace('  ', ' ').replace('  ', ' ').split(' ')
                # print(sentence_file, value_list)
                k = row[0]
                if k not in stop_words + punctuations:
                    v = np.array([float(item) for item in value_list])
                    sentence_word_embedding.append(v)
        all_sentences_word_embedding.append(sentence_word_embedding)

    # print(np.shape(all_sentences_word_embedding))  # sentence number ex. 19
    # print(np.shape(all_sentences_word_embedding[0]))  # sentence 0 words number ex. (51,768)

    # get querys embeddings path
    querys_name_set = []
    querys_embedding_set = []
    querys_embeddings_path = output_path + 'candidate_embedding/'
    with open(querys_embeddings_path + file + "_candidate_embedding.csv", newline='') as csvfile:
        spamreader = csv.reader(csvfile, delimiter=',')
        for row in spamreader:
            k = row[0]
            v = row[1].replace('\n', ' ').replace('  ', ' ').replace('  ', ' ').replace('  ', ' ')
            v = v.replace(', dtype=float32', '')[9:-3].split(']), array([')
            candidate_embeddings_set = []
            for l in range(len(v)):
                candidate_embeddings_set.append(np.array([float(item) for item in v[l].split(', ')]))
            querys_name_set.append(k)
            querys_embedding_set.append(candidate_embeddings_set)
            # print(k, candidate_embeddings_set)

    # print(len(querys_embedding_set))  # 29 querys embeddings
    # print(len(querys_embedding_set[0]))  # how many embeddings in the query0

    # main
    ranking_dict = {}
    for w in tqdm(range(len(querys_embedding_set))):
        query_inner_attn = self_attn_matrix(querys_embedding_set[w])  # shape = len(query words)*786

        sentence_embedding_set = []
        for sentence_word_embeddings in all_sentences_word_embedding:  # ex. (19, n, 768)
            cross_attn = cross_attn_matrix(sentence_word_embeddings, querys_embedding_set[w])
            sentence_embedding = self_attn_matrix(cross_attn)  # shape = (n, 768)
            sentence_embedding_set.append(sentence_embedding)  # shape = (1, 768)

        doc_inner_attn = torch.stack(sentence_embedding_set, dim=0)  # shape = (19, 768)
        doc_inner_attn = self_attn_matrix(doc_inner_attn)  # shape = (1, 768)
        output = cosine_similarity(query_inner_attn.cpu().numpy(), doc_inner_attn.cpu().numpy())
        ranking_dict[querys_name_set[w]] = float(output)

    w0 = csv.writer(open(save_path + file + '_candidate_cross_attn_value.csv', "a"))
    for k, v in sorted(ranking_dict.items(), key=lambda item:item[1], reverse=True):
        # print(k,v)
        w0.writerow([k,v])
    crt_time = time.time()
    print(n + 1, "th file", "running time", crt_time - start_time)

