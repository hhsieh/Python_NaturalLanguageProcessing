import nltk
import string
from collections import Counter
import json
import pandas as pd

data = []
with open('yelp_train_academic_dataset_review.json') as f:
    for line in f:
        data.append(json.loads(line))

text = []
for i in range(len(data)):
    text.append([data[i].values()[3]])

star = []
for i in range(len(data)):
    star.append(data[i].values()[5])

tokens = [(A.lower().replace('.', '').split(' ') for A in L) for L in text] #should be a list of lists of strings(tokens)
list_tokens_880000 = [tokens[i].next() for i in range(880000)]
#list_tokens_last = [tokens[i].next() for i in range(850000, 1012913)] #too much for the server to take in

#print(list_tokens_880000[:5])

FlattenList = sum(list_tokens_880000, [])
UniqueList = list(set(FlattenList))
CountMatrix = [[x.count(y) for y in UniqueList] for x in SplitList]
print(uniqueList)
print(CountMatrix)



#SplitList = [x[0].lower().replace('.', '').split(' ') for x in text]
#FlattenList = sum(SplitList, [])
#UniqueList = list(set(FlattenList))
#CountMatrix = [[x.count(y) for y in UniqueList] for x in SplitList]
#print(UniqueList)
#print(CountMatrix)



#print(tokens)
#print(type(tokens))
#print(tokens[0].next())
#print(tokens[1].next())

#list_tokens = [tokens[i].next() for i in range(len(tokens))]
#print(len(tokens))


#print(list_tokens[0])


#print(list(tokens[0]))
#print(list(tokens[1]))

#list_tokens = [list(tokens[i]) for i in range(len(tokens))]

#ser = pd.Series([';'.join(i) for i in tokens]).str.get_dummies(';')
#print(ser)
