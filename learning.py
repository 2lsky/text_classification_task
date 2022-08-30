from sklearn.model_selection import (GridSearchCV, 
                                    train_test_split,
                                    cross_val_score)
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import make_pipeline
from init_preproc import (X, target, stopwords, 
                        preprocess, simple_tokenizer, porter_tokenizer)
from sklearn.linear_model import LogisticRegression


vectorizer = TfidfVectorizer(strip_accents=None,
                            lowercase=False,
                            preprocessor=preprocess,
                            use_idf=True,
                            norm='l2',
                            smooth_idf=True,
                            stop_words=None,
                            tokenizer=porter_tokenizer)
clf = LogisticRegression()
X_train, X_test, y_train, y_test = train_test_split(X, 
                                                    target, 
                                                    test_size=0.2, 
                                                    random_state=0, 
                                                    shuffle=True)
estimator = make_pipeline(vectorizer, clf)
scores = cross_val_score(cv=5, 
                        estimator=estimator, 
                        verbose=4, 
                        X=X_train, 
                        y=y_train,
                        n_jobs=1)
#gs = GridSearchCV(estimator = estimator, 
                #param_grid=params, 
                #cv=10, 
                #verbose=4)
#gs.fit(X_train, y_train)
#print(gs.best_score_)
print(f'{scores.mean()} +/- {scores.std()}')