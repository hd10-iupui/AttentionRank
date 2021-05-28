import csv
import os
import time
import mxnet as mx
from bert_embedding import BertEmbedding


"""
get document word embedding by sentences
"""

ctx = mx.gpu(0)
bert = BertEmbedding(ctx=ctx)


start_time = time.time()

problem_files = []

dataset = 'SemEval2017'
text_path = dataset + '/processed_docsutf8/'
output_path = dataset + '/processed_' + dataset + '/'

files = os.listdir(text_path)
for i, file in enumerate(files):
    files[i] = file[:-4]
files = files[:]
print('docs:', len(files))

for n, file in enumerate(files):
    fp = open(text_path + file + '.txt')
    sentences = [a for a in fp.read().split('$$$$$$')]
    save_path = output_path + 'doc_word_embedding_by_sentences/' + file + '/'
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    sentences_with_embeddings = bert(sentences)  # this bert handle [list of sentences]
    for l, sentence_with_embeddings in enumerate(sentences_with_embeddings):
        words = sentence_with_embeddings[0]
        embeddings = sentence_with_embeddings[1]
        w0 = csv.writer(open(save_path + file + '_sentence' + str(l) + '_word_embeddings.csv', "a"))
        for i in range(len(words)):
            w0.writerow([words[i], embeddings[i]])

    print(n + 1, "th file", file, "running time", time.time() - start_time) #'''


