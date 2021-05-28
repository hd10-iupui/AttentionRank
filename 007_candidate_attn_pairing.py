import os
import time
import csv


"""pair candidates and their accumulated self-attention"""


dataset = 'SemEval2017'
doc_path = dataset + '/processed_docsutf8/'
output_path = dataset + '/processed_' + dataset + '/'
candidate_token_path = output_path + 'candidate_tokenizing/'
token_attn_path = output_path + 'token_attn_paired/attn/'

save_path = output_path + 'candidate_attn_paired/'
if not os.path.exists(save_path):
    os.makedirs(save_path)

files = os.listdir(doc_path)
for i, file in enumerate(files):
    files[i] = file[:-4]

files = files[:]

start_time = time.time()

for n, file in enumerate(files):

    w = csv.writer(open(save_path + file + "_attn_paired.csv", "w"))

    # read token attn to list
    token_list = []
    attn_list = []
    with open(token_attn_path + file + "token_attn_paired.csv", newline='') as csvfile:
        spamreader = csv.reader(csvfile, delimiter=',')
        for row in spamreader:
            k = row[0]
            v = float(row[1])
            token_list.append(k)
            attn_list.append(v)

    # read candidate tokens to dict
    candidate_token_dict = {}
    with open(candidate_token_path + file + "_candidate_tokenized.csv", newline='') as csvfile:
        spamreader = csv.reader(csvfile, delimiter=',')
        for row in spamreader:
            k = row[0]
            v = row[1][2:-2].split("', '")
            candidate_token_dict[k] = v

    # candidate attn pairing
    candidate_attn_dict = {}
    for k, v in candidate_token_dict.items():
        window = len(v)
        matched = []
        for t, token in enumerate(token_list):
            if token_list[t:t+window] == v:
                local_attn = sum(attn_list[t:t+window])
                if k in candidate_attn_dict.keys():
                    candidate_attn_dict[k] += local_attn
                else:
                    candidate_attn_dict[k] = local_attn

    for k, v in candidate_attn_dict.items():
        w.writerow([k, v])

    run_time = time.time()
    print(n, "th file", file, "running time", run_time - start_time)

