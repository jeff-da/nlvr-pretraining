from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np
from tqdm import tqdm

for step in range(1, 2):
    captions = []
    idx_to_line = {}

    file = open('phrase_vocab.txt')
    for i, line in enumerate(tqdm(file)):
        if i < (step * 50000) and i >= max((step - 1) * 50000, 0):
            captions.append(line.strip())
            idx_to_line[i] = line.strip()

    vect = TfidfVectorizer(min_df=1)
    tfidf = vect.fit_transform(captions)

    result = (tfidf * tfidf.T).A
    for i, array in enumerate(tqdm(result)):
        if i < (step * 50000) and i >= max((step - 1) * 50000, 0):
            arr = np.array(array)
            L = np.argsort(-arr, axis=0)
            data = idx_to_line[i]
            # print(data[0].strip() + '\t' + get_str(data[1].strip()))
            for j in range(1, 11):
                top_i = idx_to_line[L[j]]
                print(data + '\t' + top_i)
            # print('\n')