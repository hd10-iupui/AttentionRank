import csv
import os
import time
import numpy as np
import mxnet as mx
from bert_embedding import BertEmbedding
from configparser import ConfigParser
from swisscom_ai.research_keyphrase.preprocessing.postagging import PosTaggingCoreNLP
from swisscom_ai.research_keyphrase.model.input_representation import InputTextObj
from swisscom_ai.research_keyphrase.model.extractor import extract_candidates


def load_local_corenlp_pos_tagger():
    """
    terminal command:
    cd stanford-corenlp-full-2018-02-27/
    java -mx4g -cp "*" edu.stanford.nlp.pipeline.StanfordCoreNLPServer -preload tokenize,ssplit,pos -status_port 9000 -port 9000 -timeout 15000 &
    """
    config_parser = ConfigParser()
    config_parser.read('config.ini')
    host = config_parser.get('STANFORDCORENLPTAGGER', 'host')
    port = config_parser.get('STANFORDCORENLPTAGGER', 'port')
    return PosTaggingCoreNLP(host, port)


# load candidate gnerator
ptagger = load_local_corenlp_pos_tagger()

# load BERT embedding generator
ctx = mx.gpu(0)
bert = BertEmbedding(ctx=ctx, max_seq_length=64, batch_size=24)
"""if most sentences are longer than 64 words, set max_seq_length larger"""

start_time = time.time()

dataset = 'SemEval2017'
dataset_path = dataset + '/'
text_path = dataset_path + 'docsutf8/'
output_path = dataset_path + 'processed_' + dataset + '/'
if not os.path.exists(output_path):
    os.makedirs(output_path)


# read files name
files = os.listdir(text_path)
for i, file in enumerate(files):
    files[i] = file[:-4]
print(dataset, 'docs:', len(files))


# set save path for embeddings
save_path = output_path + 'candidate_embedding/'
if not os.path.exists(save_path):
    os.makedirs(save_path)


# set save path for term frequency dictionary and document frequency dictionary
df_dict = {}
save_path_tfdf = output_path + 'tf_df_dict/'
if not os.path.exists(save_path_tfdf):
    os.makedirs(save_path_tfdf)

# run all files
for n, file in enumerate(files):
    text = ''
    my_file = text_path + file + '.txt'
    with open(my_file, "r") as f:
        for line in f:
            if line:
                text += line

    tagged = ptagger.pos_tag_raw_text(text)
    text_obj = InputTextObj(tagged, 'en')
    candidates = extract_candidates(text_obj)

    w1 = csv.writer(open(save_path + file + '_candidate_embedding.csv', "a"))

    # get raw embedding
    candidates_with_embeddings = bert(candidates)  # this bert handle [list of candidates]
    for c, can_with_word_embed in enumerate(candidates_with_embeddings):

        can_words = candidates[c]  # important
        can_word_raw_embeddings = can_with_word_embed[1]
        w1.writerow([can_words, can_word_raw_embeddings])

    # get tf
    tf_dict = {}
    for item in candidates:
        item = item.lower()
        if item in tf_dict.keys():
            tf_dict[item] += 1
        else:
            tf_dict[item] = 1

    w0 = csv.writer(open(save_path_tfdf + file + '_candidate_tf.csv', "a"))
    # get df
    for k, v in sorted(tf_dict.items(), key=lambda item: item[1], reverse=True):
        w0.writerow([k, v])
        if k in df_dict.keys():
            df_dict[k] += 1
        else:
            df_dict[k] = 1

    crt_time = time.time()
    print(dataset, n + 1, "th file", file, "running time", crt_time - start_time)


# save df dictionary
w1 = csv.writer(open(save_path_tfdf + dataset + '_candidate_df.csv', "a"))
for k, v in sorted(df_dict.items(), key=lambda item:item[1], reverse=True):
    w1.writerow([k, v])


