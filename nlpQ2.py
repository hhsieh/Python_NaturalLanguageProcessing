import nltk
import string
from collections import Counter
import json
import pandas as pd
from sklearn.externals import joblib
import numpy as np
import math
from scipy.sparse import hstack

data = []
with open('yelp_train_academic_dataset_review.json') as f:
    for line in f:
        data.append(json.loads(line))


star = []
for i in range(len(data)):
    star.append(data[i].values()[5])


TEXT = [data[i].values()[3] for i in range(len(data))]

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer

vectorizer = CountVectorizer(min_df=0.01,max_df = 2.5, lowercase = False, stop_words = 'english')

X = vectorizer.fit_transform(TEXT)
transformer = TfidfTransformer()
Y = transformer.fit_transform(X)
Z = X + Y

feature_names = vectorizer.get_feature_names() 

from sklearn.linear_model import Ridge
from sklearn.cross_validation import train_test_split
from sklearn.grid_search import GridSearchCV

X_train, X_test, y_train, y_test = train_test_split(Z, star, test_size = 0.33, random_state = 42)


#alphas = np.array([50, 35, 20, 5])

#Rmodel = Ridge(alpha = 100)
#print(Rmodel.get_params().keys())

Rmodel = Ridge(alpha = 500)
#grid = GridSearchCV(estimator=Rmodel, param_grid=dict(alpha=alphas))
#gridfit = grid.fit(X_train, y_train)

#print(grid.best_score_)
#print(grid.best_estimator_.alpha)  #alpha = 100 is the best

#gridpred = grid.predict(X_test)
#gridscore = grid.score(X_test, y_test)
#print(gridscore)


Rmodelfit = Rmodel.fit(Z, star)
Rmodelpred = Rmodel.predict(Z)
Rmodelscore = Rmodel.score(Z,star)

#print(Rmodelscore)

#print(grid.best_score_)
#print(grid.best_estimator_.alpha)


from sklearn.externals import joblib
joblib.dump(Rmodel, 'nlp_q2_3.pkl')
#joblib.dump(feature_names, 'feature_names_q1.pkl')

