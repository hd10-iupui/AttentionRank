import os
import argparse
import random
import time
from bert import tokenization


"""
split document text into paired sentences, get ready to tokenize them feed bert self-attention extractor"""


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
        current_chunk.append(segment)
        current_length += len(segment)
        if i == len(document) - 1 or current_length >= target_seq_length:
            if current_chunk:
                a_end = 1
                if len(current_chunk) >= 2:
                    a_end = random.randint(1, len(current_chunk) - 1)

                tokens_a = []
                for j in range(a_end):
                    tokens_a.extend(current_chunk[j])

                tokens_b = []
                for j in range(a_end, len(current_chunk)):
                    tokens_b.extend(current_chunk[j])
                truncate_seq_pair(tokens_a, tokens_b, max_num_tokens, random)

                if len(tokens_a) == 0 or len(tokens_b) == 0:
                    break
                assert len(tokens_a) >= 1
                assert len(tokens_b) >= 1

                tokens = []
                tokens.append("[CLS]")
                for token in tokens_a:
                    tokens.append(token)
                tokens.append("[SEP]")

                for token in tokens_b:
                    tokens.append(token)
                tokens.append("[SEP]")

                instances.append(tokens)

            current_chunk = []
            current_length = 0
        i += 1
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


# main
dataset = 'SemEval2017'
text_path = dataset + '/processed_docsutf8/'
output_path = dataset + '/processed_' + dataset + '/'
save_path = output_path + 'sentence_paired_text/'

if not os.path.exists(save_path):
    os.makedirs(save_path)

bert_dir = "pretrained_bert/orgbert/"

files = os.listdir(text_path)
for i, file in enumerate(files):
    files[i] = file[:-4]

files = files[:]

start_time = time.time()

super_long = 0

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--num-docs", default=1000, type=int, help="Number of documents to use (default=1000).")
parser.add_argument("--cased", default=False, action='store_true', help="Don't lowercase the input.")
args = parser.parse_args()
tokenizer = tokenization.FullTokenizer(vocab_file=os.path.join(bert_dir, "vocab.txt"), do_lower_case=not args.cased)

for n, file in enumerate(files):
    text = ''
    my_file = text_path + file + '.txt'
    with open(my_file, "r") as f:
        for line in f:
            if line:
                text += line

    file1 = open(save_path + file + "_sentence_paired.txt", "a")

    for L in text.split('$$$$$$'):

        sub_write = 0
        line = tokenization.convert_to_unicode(L).strip()
        tokens = tokenizer.tokenize(line)
        current_doc_tokens = [tokens, tokens]

        """
        Check if the sentence will beyond max token length or not,
        if beyond, separate sentence to finer."""

        for segment in prep_document(current_doc_tokens, 512):
            # if the sentence length beyond max token number, break sentence by ';' and ','
            if (len(segment) - 1) / 2 != segment.index("[SEP]"):
                # print(segment)
                sub_write = 1
                super_long += 1
                cline = L.replace(",", ",$$$$$").replace(";", ";$$$$$").split("$$$$$")
                for part in cline:
                    # print(part)
                    file1.writelines(part)
                    file1.writelines("\n")
                    file1.writelines(part)
                    file1.writelines("\n")
                    file1.writelines("\n\n")
        # good to write
        if sub_write != 1:
            file1.writelines(L)
            file1.writelines("\n")
            file1.writelines(L)
            file1.writelines("\n")
            file1.writelines("\n\n")

    file1.close()

    run_time = time.time()
    print(n, "th file", file, "running time", run_time - start_time)

print('number of super long sentences:', super_long)