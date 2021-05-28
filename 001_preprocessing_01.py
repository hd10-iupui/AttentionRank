import os
import time
import nltk.data


"""
preprocessing document text to avoid breaking sentences and misunderstanding period
"""

tokenizer = nltk.data.load('tokenizers/punkt/english.pickle')

start_time = time.time()

path_list = ['SemEval2017']

for item in path_list:
    dataset_path = item + '/'
    text_path = dataset_path + 'docsutf8/'
    output_path = dataset_path + 'processed_' + item + '/'
    if not os.path.exists(output_path):
        os.makedirs(output_path)

    files = os.listdir(text_path)
    for i, file in enumerate(files):
        files[i] = file[:-4]

    files = files[:1]

    for n, file in enumerate(files):

        fp = open(text_path + file + '.txt')
        text = []
        for a,line in enumerate(fp):
            line = line.replace('\n','')
            if line:
                text.append(line)
        text = ''.join(text)
        # print(text)

        # replace or remove something for better sentences splitting
        text = text.replace('      ', '. ').replace('     ', '. ').replace('   ', ' ')\
            .replace('..', '.').replace(',.', ',').replace(':.', ':').replace('?.', ':')
        text = text.replace('Fig.', 'Figure').replace('Fig .', 'Figure')\
            .replace('FIG.', 'Figure').replace('FIG .', 'Figure').replace('et al.', '').replace('e.g.', '')

        # split full document text into sentences
        sentences = tokenizer.tokenize(text)
        # print(sentences)

        # double security for wrongly break a sentence into 2 pieces
        # if the heading character of a piece is lowercase, combine it to the last piece
        lower_lines = []
        for l, sentence in enumerate(sentences):
            if sentence[0].islower():
                if sentences[l-1][-1] == '.':
                    sentences[l - 1] = sentences[l-1][:-1]
                sentences[l-1] += ' ' + sentence
                lower_lines.append(l)
        # print('empty means good', lower_lines)

        final_output = []
        for l, sentence in enumerate(sentences):
            # if the heading character of a piece is lowercase, combine it to the last piece
            if l not in lower_lines and any(c.isalpha() for c in sentence):  # number_only pieces are dropped, like: '1.', '3.1.'
                final_output.append(sentence)
        # print(final_output)

        save_path = dataset_path + 'processed_docsutf8/'
        if not os.path.exists(save_path):
            os.mkdir(save_path)

        text_file = open(save_path+file+".txt", "w")
        text_file.write('$$$$$$'.join(final_output))
        text_file.close()

        print(n + 1, "th file", file, "running time", time.time() - start_time)
