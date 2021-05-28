"""Does BERT-style preprocessing of unlabeled data; heavily based on
create_pretraining_data.py in the BERT codebase. However, does not mask words
or ever use random paragraphs for the second text segment."""

import argparse
import os
import random
import time
import utils
from bert import tokenization


"""
tokenize paired sentences, get ready to feed bert self-attention extractor"""


def prep_document(document, max_sequence_length):
    """Does BERT-style preprocessing on the provided document."""
    max_num_tokens = max_sequence_length - 3
    target_seq_length = max_num_tokens

    # We DON"T just concatenate all of the tokens from a document into a long
    # sequence and choose an arbitrary split point because this would make the
    # next sentence prediction task too easy. Instead, we split the input into
    # segments "A" and "B" based on the actual "sentences" provided by the user
    # input.
    instances = []
    current_chunk = []
    current_length = 0
    i = 0
    while i < len(document):
        segment = document[i]
        # print(len(document), i, 'stop 2', segment)
        current_chunk.append(segment)
        current_length += len(segment)
        if i == len(document) - 1 or current_length >= target_seq_length:
            if current_chunk:
                # print("cc", current_chunk, len(current_chunk))
                a_end = 1
                if len(current_chunk) >= 2:
                    a_end = random.randint(1, len(current_chunk) - 1)

                tokens_a = []
                for j in range(a_end):
                    tokens_a.extend(current_chunk[j])
                # print("aa", tokens_a)

                tokens_b = []
                # print("lc", len(current_chunk))
                for j in range(a_end, len(current_chunk)):
                    # print("j=", j)
                    tokens_b.extend(current_chunk[j])
                truncate_seq_pair(tokens_a, tokens_b, max_num_tokens, random)
                # print("bb", tokens_b)

                if len(tokens_a) == 0 or len(tokens_b) == 0:
                    break
                assert len(tokens_a) >= 1
                assert len(tokens_b) >= 1

                tokens = []
                tokens.append("[CLS]")
                for token in tokens_a:
                    tokens.append(token)
                # print("cs", tokens)
                tokens.append("[SEP]")

                for token in tokens_b:
                    tokens.append(token)
                tokens.append("[SEP]")

                instances.append(tokens)

            current_chunk = []
            current_length = 0
        i += 1
    # print(instances)
    return instances


def truncate_seq_pair(tokens_a, tokens_b, max_num_tokens, rng):
    """Truncates a pair of sequences to a maximum sequence length."""
    while True:
        total_length = len(tokens_a) + len(tokens_b)
        if total_length <= max_num_tokens:
            break

        trunc_tokens = tokens_a if len(tokens_a) > len(tokens_b) else tokens_b
        assert len(trunc_tokens) >= 1

        # We want to sometimes truncate from the front and sometimes from the
        # back to add more randomness and avoid biases.
        if rng.random() < 0.5:
            del trunc_tokens[0]
        else:
            trunc_tokens.pop()


dataset = 'SemEval2017'
text_path = dataset + '/processed_docsutf8/'
output_path = dataset + '/processed_' + dataset + '/'
save_path = output_path + 'sentence_paired_text/'

# set which pre-trained bert is going to use
bert_dir = "pretrained_bert/orgbert/"
bert_name = bert_dir.split('/')[1]

files = os.listdir(text_path)
for i, file in enumerate(files):
    files[i] = file[:-4]

files = files[:]

start_time = time.time()
parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--num-docs", default=1000, type=int,help="Number of documents to use (default=1000).")
parser.add_argument("--cased", default=False, action='store_true',help="Don't lowercase the input.")
parser.add_argument("--max_sequence_length", default=512, type=int, help="Maximum input sequence length after tokenization (default=128).")
args = parser.parse_args()
tokenizer = tokenization.FullTokenizer(vocab_file=os.path.join(bert_dir, "vocab.txt"), do_lower_case=not args.cased)

for n, file in enumerate(files):
    data_file = save_path + file + '_sentence_paired.txt'
    random.seed(0)
    current_doc_tokens = []
    segments = []

    with open(data_file, "r") as f:
        raw_lines = []
        for l, line in enumerate(f):
            raw_lines.append(line)
            line = tokenization.convert_to_unicode(line).strip()
            # print(line)
            if not line:  # line is empty, deal the 2 segments in the [current doc tokens]
                if current_doc_tokens:

                    for segment in prep_document(
                            current_doc_tokens, args.max_sequence_length):
                        if (len(segment) - 1) / 2 != segment.index("[SEP]"):
                            # print("0", segment.index("[SEP]"), len(segment) - 1)
                            # print(l, segment)
                            cline = raw_lines[l - 1].replace("),", "),$$$$$").split("$$$$$")
                            # for part in cline:
                            # print(part)
                        # print(segment)
                        segments.append(segment)
                        if len(segments) >= args.num_docs:
                            break
                    if len(segments) >= args.num_docs:
                        break

                current_doc_tokens = []  # clean up [current doc tokens]

            tokens = tokenizer.tokenize(line)
            if tokens:  # if line is not empty, add lines to [current doc tokens]
                current_doc_tokens.append(tokens)
    run_time = time.time()
    print(n, "th file", file, "running time", run_time - start_time)
    utils.write_json([{"tokens": s} for s in segments],
                     data_file.replace("_sentence_paired.txt", "")+"_"+bert_name+".json")
