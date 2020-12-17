import sys
import numpy as np
import pandas as pd
from sqlalchemy import create_engine
import sqlite3
import nltk
import re
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer,FeatureHasher
from sklearn.preprocessing import OneHotEncoder,LabelEncoder
from sklearn.datasets import make_multilabel_classification
from sklearn.multioutput import MultiOutputClassifier
from sklearn.metrics import classification_report,accuracy_score,homogeneity_score,completeness_score,f1_score,precision_score,recall_score
from sklearn.model_selection import GridSearchCV
from sklearn.base import TransformerMixin, BaseEstimator
from sklearn.decomposition import LatentDirichletAllocation, NMF
from sklearn.externals import joblib

nltk.download('punkt')
nltk.download('wordnet')
nltk.download('stopwords')


def load_data(database_filepath):
    """
    Load the databaase and return the features, label and the category names
    """
    con = sqlite3.connect(database_filepath)
    df = pd.read_sql("SELECT * FROM DisasterResponse", con)
    X = df[['message','genre']].values
    y = df.iloc[:,4:].values
    category_names = df.columns[4:]
    return X, y, category_names

def tokenize(text):
    """
    Tokenize a text message into a list of processed word

    Input:
    text: str; original text message

    Return:
    tokens: list; list of processed word
    """
    text = re.sub(r"[^a-zA-Z0-9]", " ", text.lower())
    tokens = nltk.word_tokenize(text)
    # tokens = [tok for tok in tokens if tok not in nltk.corpus.stopwords.words("english")]
    tokens = [nltk.stem.WordNetLemmatizer().lemmatize(tok).lower().strip() for tok in tokens]
    return tokens

class Column_extractor(TransformerMixin, BaseEstimator):
    """
    A transformer that returns array of the predefined column(s) from the input array
    """
    def __init__(self,col_nums,vectorise=True):
        self.col_nums = col_nums
        self.vectorise = vectorise

    def fit(self,X,y=None):
        return self

    def transform(self, X):
        if self.vectorise == True:
            return X[:,self.col_nums].reshape(-1,1)
        else:
            return X[:,self.col_nums]


def transfrom_text_query(query):
    """
    transform text into the input format for the model

    Input:
    query: str; the text that is passing to the web apps, user can defined the
    genre of the message by adding "|", the avalable genre are 'direct',
    'social', 'news'

    Output:
    arr: Numpy.ndarry; correct format for inputing into the model

    """
    items = query.split('|')
    if len(items) == 1:
        arr = np.array([items[0],'direct']).reshape(1,2)
    else:
        arr =  np.array(items).reshape(1,len(items))
    return arr

def build_pipeline():
    """
    Return data preprocessing Pipeline
    """

    nlp_pipeline = Pipeline([
        ('text_extract',Column_extractor(0,vectorise=False)),
        ('vect', CountVectorizer(tokenizer=tokenize)),
        ('tfidf', TfidfTransformer()),
#         ('nmf',NMF(n_components=36,random_state=42))
#         ('lda',LatentDirichletAllocation(n_components=36,random_state=42))

    ])
    genre_pipline = Pipeline([
        ('genre_extract', Column_extractor(1,vectorise=False)),
        ('encoding',CountVectorizer()),
    ])
    pipeline = Pipeline([
            ('features', FeatureUnion([
                ('content', nlp_pipeline),
                ('genre', genre_pipline),
            ])),
        ])
    return pipeline

def build_model(model_func=None,search=False):
    """
    Build model (default:RandomForestClassifier)

    Input:
    model_func: sklearn.base; a sklearn model
    search: bool; applying GridSearchCV

    Output:
    model: sklearn.base; model
    """

    if model_func == True:
        model = model_func()
    else:
        model = MultiOutputClassifier(RandomForestClassifier())

    if search == True:
        parameters = {
            'estimator__n_estimators': [50,100,200],
            'estimator__min_samples_split': [2,3,4]
        }
        cv = GridSearchCV(model, param_grid=parameters)
        print("GridSearchCV model built")
        return cv
    else:
        print("Normal model built")
        return model


def evaluate_model(pipeline, model, X_test, y_test, category_names, search=None):
    """
    Validate model (and pipeline) using test setting

    Input:
    pipeline: sklearn.pipeline; pipeline for preprocessing the data, data
    will not be transformed if it is not equal to True
    model: sklearn.base; trained model for perdiction
    X_test:
    y_test:
    category_names:
    search: bool: if True, it prints the best parameter of the model

    """
    assert y_test.shape[0] == X_test.shape[0]
    X_test = pipeline.transform(X_test )
    y_pred = model.predict(X_test)
    assert y_test.shape == y_pred.shape
    scores = []
    for i in range(y_pred.shape[-1]):
        precision = precision_score(y_test[:,i],y_pred[:,i],average='macro')
        recall = recall_score(y_test[:,i],y_pred[:,i],average='macro')
        f1 =  f1_score(y_test[:,i],y_pred[:,i],average='macro')
        print('category: ',category_names[i],'\tprecision: ',round(precision),'\trecall: ',round(recall),'\tf1: ',round(f1))
    if search == True:
        print("Best Parameters:", model.best_params_)
    return


def save_model(model, model_filepath):
    joblib.dump(model, model_filepath)
    return


def main():
    if 3 <= len(sys.argv) <= 4:

        # input
        tranlator = lambda x: True if x == "True" else False
        if len(sys.argv) == 3:
            database_filepath, model_filepath = sys.argv[1:]
            search = False
        else:
            database_filepath, model_filepath, search = sys.argv[1:]
        search = tranlator(search)

        # load data
        print('Loading data...\n    DATABASE: {}'.format(database_filepath))
        X, Y, category_names = load_data(database_filepath)
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)

        print('Building pipeline...')
        pipeline = build_pipeline()

        print('Building model...')
        model = build_model(model_func=None, search=search)

        print('Transform data...')
        X_train = pipeline.fit_transform(X_train)

        print('Training model...')
        model.fit(X_train, Y_train)

        print('Evaluating model...')
        evaluate_model(pipeline, model, X_test, Y_test, category_names,search)

        print('Saving models...')
        save_model(pipeline, model_filepath +'/pipeline.pkl')
        save_model(model, model_filepath +'/classifier.pkl')

        print('Trained model saved!')

    else:
        print('Please provide the filepath of the disaster messages database '\
              'as the first argument and the filepath of the pickle file to '\
              'save the model to as the second argument. \n\nExample: python '\
              'train_classifier.py ../data/DisasterResponse.db classifier.pkl')


if __name__ == '__main__':
    main()
