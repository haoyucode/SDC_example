# Classify Social Media Posts (binary classification)
# This template classifies whether a tweet is commercial

import os
import pandas as pd
from sklearn.metrics import confusion_matrix
import joblib
from sklearn.feature_extraction.text import TfidfTransformer, CountVectorizer
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import classification_report
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import cross_val_predict
from sklearn.pipeline import FeatureUnion
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split

def getKey(item):
    return item[1]

# Load data
df = pd.read_csv(r'path to human labeled social media posts.csv')

# Split data
full_train, full_test = train_test_split(full_df, test_size=0.2, random_state=233)

### DEFAULT PIPELINE
pipeline = Pipeline([
        ('union', FeatureUnion(
            transformer_list=[
                # Extracting ngrams
                ('ngram', Pipeline([
                    ('text_vect', CountVectorizer(
                        analyzer='char',
                        ngram_range=(3, 5),
                        min_df =2,
                        strip_accents=None,
                        decode_error='replace')
                     ),
                    ('tfidf', TfidfTransformer()),
                    ('best', SelectKBest(chi2, k=10000)),
                ])),
                # Extracting bag of word
                ('word', Pipeline([
                    ('text_vect', CountVectorizer(
                        analyzer='word',
                        ngram_range=(1, 1),
                        stop_words='english',
                        strip_accents=None,
                        decode_error='replace')
                     ),
                    ('tfidf', TfidfTransformer()),
                    ('best', SelectKBest(chi2, k=1000)),
                ])),
            ],
            # weight components in FeatureUnion
            transformer_weights={
                'ngram': 1.0,
                'word': 1.0,
            },
        )),
        # Classification model
        ('clf',
         SGDClassifier(max_iter=200, n_jobs=4, penalty='elasticnet')
         ),
    ])

### GRID SEARCH
# Defining parameter grids for searching
union_grid = dict(
    union__ngram__best__k=[5000, 10000, 15000, 20000],
    union__ngram__text_vect__ngram_range=[(2, 4), (2, 5), (3, 3), (3, 4), (3, 5), (4, 4), (4, 5)], #new
    union__word__best__k=[1000, 2000, 3000, 5000],
    clf__loss=['hinge', 'log', 'modified_huber'],
    # clf__loss__penalty=['elasticnet', 'l1', 'l2'],  # new
    clf__l1_ratio=[0.0, 0.25, 0.5, 0.75, 1.0],
    clf__class_weight=[None, 'balanced'],
    clf__alpha=[0.001, 0.0001, 0.00001, 0.000001]
)
# Grid search
grid_search_union = GridSearchCV(pipeline, scoring='f1',
                                 param_grid=union_grid, cv=5, verbose=10,
                                 n_jobs=3)

grid_search_union.fit(full_train['body_master'], full_train['Commercial'])

# Storing the scores
grid_scores_union = grid_search_union.cv_results_
sorted_scores = sorted(grid_scores_union, key=getKey, reverse=True)
# Storing the best model
best_est = grid_search_union.best_estimator_

print(best_est.steps)

final_model = best_est.fit(full_train['body_master'], full_train['Commercial'])

# Classifier Performance
# On training set
cv_pred = cross_val_predict(final_model, full_train['body_master'], full_train['Commercial'], cv=10)

print(classification_report(full_train['Commercial'].values, cv_pred))
# Report on training set - CV
#               precision    recall  f1-score   support
#          0.0       0.92      0.89      0.91      1715
#          1.0       0.91      0.93      0.92      1881
#     accuracy                           0.91      3596
#    macro avg       0.91      0.91      0.91      3596
# weighted avg       0.91      0.91      0.91      3596

# On test set
full_test['pred_commercial'] = final_model.predict(full_test['body_master'])
clf_rpt_on_test = cross_val_predict(best_est, full_test['body_master'], full_test['Commercial'].values, cv=10)
file = open(r'path to save evaluation results.txt', 'w')
file.write(classification_report(full_test['Commercial'].values, clf_rpt_on_test))
file.close()

# Report on test set
#               precision    recall  f1-score   support
#          0.0       0.94      0.93      0.93       400
#          1.0       0.94      0.95      0.95       500
#     accuracy                           0.94       900
#    macro avg       0.94      0.94      0.94       900
# weighted avg       0.94      0.94      0.94       900

joblib.dump(best_est, r'path to save the best model.pkl')
