import json
import pandas as pd
import numpy as np

data = []

with open('/home/vagrant/miniprojects/ml/yelp_data.json') as f:
    for line in f:
        data.append(json.loads(line))

categories = []
for i in range(len(data)):
        categories.append(data[i].values()[14])

bid = []
for i in range(len(data)):
        bid.append(data[i].values()[5])

review = []
with open("yelp_train_academic_dataset_review.json") as f:
    for line in f:
        review.append(json.loads(line))

rid = []
for i in range(len(review)):
    rid.append(review[i].values()[4])

text = []
for i in range(len(review)):
    text.append(review[i].values()[3])

star = []
for i in range(len(review)):
    star.append(review[i].values()[5])

new = pd.DataFrame({"rid": rid, "text": text, "star": star})

ser = pd.Series([';'.join(i) for i in categories]).str.get_dummies(';')
ser_columns = ser.columns

Restaurants = ser["Restaurants"]

dd = pd.DataFrame({"bid" :bid, "Restaurant": Restaurants})
selected_id = dd.loc[dd["Restaurant"] == 1]
selected_id = selected_id["bid"]


filtered = new[new["rid"].isin(selected_id)]  #this is what you need
#print(filtered)

TEXT = filtered["text"]

from sklearn.feature_extraction.text import CountVectorizer
#from sklearn.feature_extraction.text import TfidfTransformer

bigram_vectorizer = CountVectorizer(ngram_range=(1,2), token_pattern=r'\b\w+\b', min_df=0.01,max_df = 2.5, lowercase = False, stop_words = 'english')


X = bigram_vectorizer.fit_transform(TEXT)
feature_names = bigram_vectorizer.get_feature_names()

matrix_terms = np.array(feature_names)
matrix_freq = np.asarray(X.sum(axis=0)).ravel()
matrix_freq = matrix_freq.astype(np.float)

final_matrix = np.array([matrix_terms, matrix_freq])
sum_count = sum(matrix_freq)
w_ratio = matrix_freq/sum_count
#print(matrix_terms, w_ratio)

#print(feature_names[:50])
#print(feature_names[-50:-1])
#print(len(feature_names))
#print(w_ratio[:50])
#print(w_ratio[-50:-1])
#print(len(w_ratio))


S = feature_names
P = w_ratio
for i in range(len(S)):
    s_split = S[i].split()
    s_split_len = len(S[i].split())
    if s_split_len == 2:
        a = S.index(S[i])
        b = S.index(S[i].split()[0])
        c = S.index(S[i].split()[1])
        if a != None:
            if b != None:
                co = [a, b, c]
                probs = P[co[0]], P[co[1]], P[co[2]]
                prob = probs[0] % (probs[1] * probs[2])
                topbi = [S[i], probs[0] % (probs[1] * probs[2])]
                print topbi          
